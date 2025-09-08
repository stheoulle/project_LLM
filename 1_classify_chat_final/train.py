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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
except Exception:
    raise RuntimeError('torch is required for training. Install with pip install torch')

from model_backbones import get_backbone
from model_arch import GlobalLocalModel


class NumpyImageDataset(Dataset):
    def __init__(self, root: str, transform=None, infer_labels: bool = True, labels_df=None, labels_col: str = 'Pathology'):
        self.root = Path(root)
        # only consider top-level preprocessed full images (exclude patch files and patch folders)
        all_npy = list(self.root.rglob('*.npy'))
        files = []
        for p in all_npy:
            # skip files inside a folder that ends with _patches
            if any(part.endswith('_patches') for part in p.parts):
                continue
            # skip files named patch_*.npy
            if p.name.startswith('patch_'):
                continue
            files.append(p)
        self.files = files
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

        # gather patches if available: look for sibling folder <stem>_patches
        patches_dir = p.parent / f"{p.stem}_patches"
        patches = None
        if patches_dir.exists() and patches_dir.is_dir():
            patch_files = sorted(patches_dir.glob('*.npy'))
            if patch_files:
                patches = [np.load(pp) for pp in patch_files]
                # ensure channel-first for patches
                patches = [((pp if pp.ndim == 3 and pp.shape[0] == 1 else (pp[np.newaxis, ...] if pp.ndim==2 else np.transpose(pp,(2,0,1))))).astype(np.float32) for pp in patches]
        return arr, np.float32(label), patches


def collate_batch(batch):
    """Collate a batch of (global_image, label, patches_list) tuples.
    Pads global images in the batch to the same H,W and stacks.
    Aggregates patches across the batch into a single tensor and returns patch_counts per sample.
    """
    import torch
    import torch.nn.functional as F

    globals_list = [torch.from_numpy(b[0]) for b in batch]
    labels = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))
    patches_list = [b[2] for b in batch]

    # pad globals to same H,W
    max_h = max(img.shape[1] for img in globals_list)
    max_w = max(img.shape[2] for img in globals_list)
    padded_globals = []
    for img in globals_list:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        pad = (0, pad_w, 0, pad_h)
        img_p = img if (pad_h == 0 and pad_w == 0) else F.pad(img, pad, value=0.0)
        padded_globals.append(img_p)
    globals_tensor = torch.stack(padded_globals, dim=0)

    # aggregate patches
    have_patches = any(pl is not None and len(pl) > 0 for pl in patches_list)
    if not have_patches:
        return globals_tensor, labels, None, [0] * len(batch)

    # flatten all patches and record counts per sample
    all_patches = []
    patch_counts = []
    for pl in patches_list:
        if pl is None or len(pl) == 0:
            patch_counts.append(0)
            continue
        patch_counts.append(len(pl))
        for pp in pl:
            all_patches.append(torch.from_numpy(pp))

    # pad patches to max patch H,W
    max_ph = max(p.shape[1] for p in all_patches)
    max_pw = max(p.shape[2] for p in all_patches)
    padded = []
    for p in all_patches:
        c, h, w = p.shape
        pad_h = max_ph - h
        pad_w = max_pw - w
        pad = (0, pad_w, 0, pad_h)
        p_p = p if (pad_h == 0 and pad_w == 0) else F.pad(p, pad, value=0.0)
        padded.append(p_p)
    patches_tensor = torch.stack(padded, dim=0) if padded else None

    return globals_tensor, labels, patches_tensor, patch_counts


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
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    labels_df = None
    if labels_excel is not None:
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError('pandas is required to read Excel labels. Install with: pip install pandas openpyxl')
        try:
            labels_df = pd.read_excel(labels_excel)
            print(f'Loaded labels from {labels_excel}, shape={labels_df.shape}')
        except Exception as e:
            raise RuntimeError(f'Failed to read labels Excel: {e}')

    ds = NumpyImageDataset(preprocessed_root, labels_df=labels_df, labels_col=labels_col)
    if max_samples is not None:
        # subsample
        ds.files = ds.files[:max_samples]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # create GlobalLocalModel
    model = GlobalLocalModel(global_backbone=model_name, pretrained=True, in_channels=1, radimagenet_path=radimagenet_path, share_local_global=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels, patches, patch_counts in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if patches is not None:
                patches = patches.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                logits = model(imgs, patches=patches, patch_counts=patch_counts)
                logits = logits.view(-1)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.shape[0]
            total += imgs.shape[0]
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()

        avg_loss = running_loss / max(1, total)
        acc = correct / max(1, total)
        print(f'Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} acc: {acc:.4f}')

    # save final model
    out_dir = Path(preprocessed_root) / '..' / 'training_out'
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, out_dir / f'{model_name}_final.pth')
    print('Training finished. Checkpoint saved to', out_dir)


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
