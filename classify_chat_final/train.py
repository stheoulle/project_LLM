"""Simple training runner for preprocessed mammography data.

This is a lightweight trainer to get you started. It expects a directory containing preprocessed .npy files
and optionally subfolders with patches. The labels are not provided here; for demo purposes the trainer
looks for filenames containing 'pos' or 'neg' to infer labels. Provide a CSV labels file for real training.

Usage:
    from train import run_training
    run_training(preprocessed_root, model_name='resnet50', epochs=5, batch_size=8, lr=1e-4)

Dependencies: torch, numpy, tqdm
"""
from pathlib import Path
import numpy as np
import os
import glob
import json
from typing import Optional
from contextlib import nullcontext
import sys
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
except Exception:
    raise RuntimeError('torch is required for training. Install with pip install torch')

from model_backbones import get_backbone


class NumpyImageDataset(Dataset):
    def __init__(self, root: str, transform=None, infer_labels: bool = True, labels_df=None, labels_col: str = 'Pathology'):
        self.root = Path(root)
        self.files = list(self.root.rglob('*.npy'))
        self.transform = transform
        self.infer_labels = infer_labels
        self.labels_df = labels_df
        self.labels_col = labels_col
        if not self.files:
            raise RuntimeError(f'No .npy files found under {root}')

        # build simple lookup if labels_df provided
        if self.labels_df is not None:
            # normalize df for fast search
            self._df = self.labels_df.astype(str).apply(lambda x: x.str.lower())
        else:
            self._df = None

    def __len__(self):
        return len(self.files)

    def _infer_label_from_df(self, sample_path: Path):
        # try to read accompanying .json metadata produced by preprocess.py
        candidate_keys = []
        json_path = sample_path.with_suffix('.json')
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
                orig = meta.get('orig_path') or meta.get('metadata', {}).get('path')
                if orig:
                    candidate_keys.append(Path(orig).name.lower())
                    candidate_keys.append(Path(orig).stem.lower())
            except Exception:
                pass
        # also use the npy filename
        candidate_keys.append(sample_path.name.lower())
        candidate_keys.append(sample_path.stem.lower())

        # search for any matching row in dataframe: check if any cell contains the key substring
        for key in candidate_keys:
            for col in self._df.columns:
                mask = self._df[col].str.contains(key, na=False)
                if mask.any():
                    # take first match
                    row = self.labels_df.loc[mask.idxmax()]
                    val = row.get(self.labels_col)
                    return val
        return None

    def __getitem__(self, idx):
        p = self.files[idx]
        arr = np.load(p)
        # ensure channel first
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[0] != 1 and arr.shape[-1] != 1:
            # assume HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr.astype(np.float32)
        # infer label (very naive): filename contains 'pos' or 'neg' or 'malig' / 'benign'
        label = 0.0
        if self.infer_labels:
            assigned = None
            if self._df is not None:
                try:
                    val = self._infer_label_from_df(Path(p))
                    if val is not None:
                        assigned = val
                except Exception:
                    assigned = None
            if assigned is None:
                n = p.name.lower()
                if 'pos' in n or 'malig' in n or 'malignant' in n or 'cancer' in n:
                    label = 1.0
                elif 'benign' in n or 'neg' in n or 'normal' in n:
                    label = 0.0
                elif '1' in n and '0' not in n:
                    label = 1.0
                else:
                    label = 0.0
            else:
                # map assigned value to binary
                s = str(assigned).lower()
                if any(k in s for k in ['malig', 'malignant', 'cancer', 'positive', 'pos', '1']):
                    label = 1.0
                else:
                    label = 0.0
        return arr, np.float32(label)


def collate_batch(batch):
    """Collate a batch of (numpy_image, label) pairs.
    Pads all images in the batch to the same spatial size (max H,W) with zeros and stacks into a tensor.
    """
    import torch
    import torch.nn.functional as F

    imgs = [torch.from_numpy(b[0]) for b in batch]
    labels = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))

    # determine max height and width in this batch
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)

    padded = []
    for img in imgs:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad format for F.pad: (pad_left, pad_right, pad_top, pad_bottom)
        pad = (0, pad_w, 0, pad_h)
        if pad_h == 0 and pad_w == 0:
            img_p = img
        else:
            img_p = F.pad(img, pad, value=0.0)
        padded.append(img_p)

    imgs = torch.stack(padded, dim=0)
    return imgs, labels


def run_training(preprocessed_root: str,
                 model_name: str = 'resnet50',
                 epochs: int = 5,
                 batch_size: int = 8,
                 lr: float = 1e-4,
                 device: Optional[str] = None,
                 radimagenet_path: Optional[str] = None,
                 max_samples: Optional[int] = None,
                 labels_excel: Optional[str] = None,
                 labels_col: str = 'Pathology'):
    preprocessed_root = str(preprocessed_root)

    # Safe device detection with graceful fallback if CUDA/NVML initialization fails
    if device is None:
        # try to detect CUDA but be robust against NVML / driver errors
        try:
            try:
                cuda_available = torch.cuda.is_available()
            except Exception as e:
                print('Warning: torch.cuda.is_available() raised:', e, file=sys.stderr, flush=True)
                cuda_available = False

            if cuda_available:
                try:
                    # probe current device to force initialization and catch errors
                    torch.cuda.current_device()
                    _ = torch.cuda.device_count()
                    device = 'cuda'
                except Exception as e:
                    print('Warning: CUDA initialization failed, falling back to CPU:', e, file=sys.stderr, flush=True)
                    device = 'cpu'
            else:
                device = 'cpu'
        except Exception:
            device = 'cpu'
    else:
        # user provided device: validate it
        if device == 'cuda':
            try:
                torch.cuda.current_device()
            except Exception as e:
                print('Warning: requested CUDA device unavailable, falling back to CPU:', e, file=sys.stderr, flush=True)
                device = 'cpu'

    print('Using device:', device, file=sys.stderr, flush=True)

    labels_df = None
    if labels_excel is not None:
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError('pandas is required to read Excel labels. Install with: pip install pandas openpyxl')
        try:
            labels_df = pd.read_excel(labels_excel)
            print(f'Loaded labels from {labels_excel}, shape={labels_df.shape}', file=sys.stderr, flush=True)
        except Exception as e:
            raise RuntimeError(f'Failed to read labels Excel: {e}')

    ds = NumpyImageDataset(preprocessed_root, labels_df=labels_df, labels_col=labels_col)
    if max_samples is not None:
        # subsample
        ds.files = ds.files[:max_samples]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    print(f'Dataset size: {len(ds)} files, batch_size: {batch_size}', file=sys.stderr, flush=True)
    batches_per_epoch = len(dl)
    print(f'Batches per epoch: {batches_per_epoch}', file=sys.stderr, flush=True)
    print(f'Model: {model_name}, lr: {lr}, epochs: {epochs}', file=sys.stderr, flush=True)

    model, feat_dim = get_backbone(name=model_name, pretrained=True, in_channels=1, radimagenet_path=radimagenet_path)
    model = model.to(device)
    # append a simple classification head
    head = nn.Linear(feat_dim, 1)
    head = head.to(device)

    optimizer = AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    head.train()

    # prepare amp context manager (use nullcontext when running on CPU)
    amp_ctx = (torch.amp.autocast('cuda') if device != 'cpu' else nullcontext())

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}', file=sys.stderr, flush=True)
        running_loss = 0.0
        total = 0
        correct = 0

        # Use tqdm to show progress per batch for UI (write to stderr so it appears in terminal)
        batch_iter = tqdm(dl, desc=f'Epoch {epoch}/{epochs}', leave=False, unit='batch', file=sys.stderr)
        for batch_idx, (imgs, labels) in enumerate(batch_iter, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # choose amp context based on current device (re-evaluated each iteration so fallback works)
            amp_ctx = (torch.amp.autocast('cuda') if device != 'cpu' else nullcontext())

            try:
                with amp_ctx:
                    feats = model.forward_features(imgs) if hasattr(model, 'forward_features') else model(imgs)
                    # some timm models return (B, feat_dim, 1, 1) or (B, feat_dim)
                    if feats.ndim > 2:
                        feats = feats.view(feats.shape[0], -1)
                    logits = head(feats).view(-1)
                    loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            except Exception as e:
                # detect NVML / CUDACachingAllocator style failures and fallback to CPU
                msg = str(e).lower()
                if any(k in msg for k in ('nvml', 'cudacachingallocator', 'driver/library', 'nvml_success')):
                    print('Warning: CUDA runtime error detected during forward (likely driver/library mismatch):', e, file=sys.stderr, flush=True)
                    print('Falling back to CPU for remainder of training.', file=sys.stderr, flush=True)
                    device = 'cpu'
                    # move model and head to CPU
                    model = model.to(device)
                    head = head.to(device)
                    # move current batch to cpu
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    # recompute on CPU without amp
                    with nullcontext():
                        feats = model.forward_features(imgs) if hasattr(model, 'forward_features') else model(imgs)
                        if feats.ndim > 2:
                            feats = feats.view(feats.shape[0], -1)
                        logits = head(feats).view(-1)
                        loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    # unknown error -> re-raise
                    raise

            batch_loss = loss.item()
            running_loss += batch_loss * imgs.shape[0]
            total += imgs.shape[0]
            batch_correct = ( (torch.sigmoid(logits) > 0.5).float() == labels ).sum().item()
            correct += batch_correct
            batch_acc = batch_correct / imgs.shape[0]

            # update tqdm postfix
            batch_iter.set_postfix({'batch': f'{batch_idx}/{batches_per_epoch}', 'loss': f'{batch_loss:.4f}', 'batch_acc': f'{batch_acc:.3f}'})

        avg_loss = running_loss / max(1, total)
        acc = correct / max(1, total)
        print(f'Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} acc: {acc:.4f}', file=sys.stderr, flush=True)

    # save final model
    out_dir = Path(preprocessed_root) / '..' / 'training_out'
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'head_state_dict': head.state_dict()}, out_dir / f'{model_name}_final.pth')
    print('Training finished. Checkpoint saved to', out_dir, file=sys.stderr, flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--preprocessed_root', required=True)
    p.add_argument('--model_name', default='resnet50')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--labels_excel', type=str, default=None)
    p.add_argument('--labels_col', type=str, default='Pathology')
    p.add_argument('--max_samples', type=int, default=0)
    args = p.parse_args()
    run_training(args.preprocessed_root, model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, labels_excel=args.labels_excel, labels_col=args.labels_col, max_samples=(args.max_samples or None))
