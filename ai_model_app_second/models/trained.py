import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm

# === New imports for CBIS-DDSM integration ===
import os
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as T
from collections import defaultdict, Counter


def resize_volume(volume, target_shape=(128, 224, 224)):
    # volume: [C, D, H, W]
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=target_shape, mode='trilinear', align_corners=False)
    return resized.squeeze(0)  # [C, D, H, W]


# === New: Simple 2D image resize/normalize for mammograms ===
def build_image_transform(img_size=224, train=False):
    # Train-time light augmentations to reduce overfitting/collapse
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.ToTensor(),                 # [1, H, W] in [0,1]
            T.Normalize(mean=[0.5], std=[0.25]),
        ])
    # Eval-time deterministic preprocessing
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((img_size, img_size)),
        T.ToTensor(),                 # [1, H, W] in [0,1]
        T.Normalize(mean=[0.5], std=[0.25]),
    ])


# === New: Utilities to map CBIS-DDSM CSV rows to JPEG images via dicom_info.csv ===
class CBISIndex:
    """Index dicom_info.csv once and enable quick lookups by PatientName and SeriesDescription."""
    def __init__(self, csv_dir: str):
        self.df = pd.read_csv(os.path.join(csv_dir, 'dicom_info.csv'))
        # Normalize columns
        if 'PatientName' not in self.df.columns:
            raise RuntimeError('dicom_info.csv missing PatientName column')
        self.df['SeriesDescription'] = self.df.get('SeriesDescription', pd.Series(index=self.df.index, dtype=str)).fillna('')
        # Build index: (PatientName, SeriesDescription) -> list[jpeg_path]
        self.by_name_desc = defaultdict(list)
        for _, row in self.df.iterrows():
            name = str(row['PatientName'])
            desc = str(row['SeriesDescription'])
            img_path = str(row['image_path']) if 'image_path' in row else None
            if img_path and img_path.strip():
                self.by_name_desc[(name, desc)].append(img_path)
        # Also fallback index only by PatientName
        self.by_name = defaultdict(list)
        for (name, desc), lst in self.by_name_desc.items():
            self.by_name[name].extend(lst)

    def find_jpegs(self, patient_name: str, prefer=('cropped images', 'full mammogram images', 'ROI mask images')):
        # Try preferred series in order
        for desc in prefer:
            key = (patient_name, desc)
            if key in self.by_name_desc and len(self.by_name_desc[key]) > 0:
                return self.by_name_desc[key]
        # Fallback to any images for that name
        return self.by_name.get(patient_name, [])


# === New: Tabular encoders for categorical metadata ===
class OneHotEncoder:
    def __init__(self, unknown_token='__UNK__'):
        self.unknown = unknown_token
        self.vocab = {self.unknown: 0}

    def fit(self, values):
        for v in values:
            v = self._norm(v)
            if v not in self.vocab:
                self.vocab[v] = len(self.vocab)

    def _norm(self, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return self.unknown
        return str(v).strip().upper()

    @property
    def dim(self):
        return len(self.vocab)

    def transform(self, v):
        one = np.zeros(self.dim, dtype=np.float32)
        idx = self.vocab.get(self._norm(v), 0)
        one[idx] = 1.0
        return one

def safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) and val != '' else default
    except (ValueError, TypeError):
        return default


class CBISDDSMCaseDataset(Dataset):
    """
    Case-level dataset built from CBIS-DDSM calc/mass description CSVs.
    Each row is a case and yields (image_tensor, tabular_tensor, label).

    Label mapping: BENIGN and BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        use_cropped: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        assert split in ('train', 'test'), "split must be 'train' or 'test'"
        self.base_dir = base_dir
        self.csv_dir = os.path.join(base_dir, 'csv')
        self.jpeg_root = os.path.join(base_dir, 'jpeg')  # paths in dicom_info are relative to repo root
        self.index = CBISIndex(self.csv_dir)
        self.transform = build_image_transform(img_size, train=(split == 'train'))

        # Load description CSVs
        calc_csv = os.path.join(self.csv_dir, f'calc_case_description_{"train" if split=="train" else "test"}_set.csv')
        mass_csv = os.path.join(self.csv_dir, f'mass_case_description_{"train" if split=="train" else "test"}_set.csv')
        df_list = []
        if os.path.exists(calc_csv):
            df = pd.read_csv(calc_csv)
            df['__modality__'] = 'CALC'
            df_list.append(df)
            print(f"Loaded calc cases from {calc_csv}")
        else:
            print(f"Warning: {calc_csv} not found, skipping calc cases.")
        if os.path.exists(mass_csv):
            df = pd.read_csv(mass_csv)
            df['__modality__'] = 'MASS'
            df_list.append(df)
            print(f"Loaded mass cases from {mass_csv}")
        if not df_list:
            raise RuntimeError(f"No case description CSVs found in {self.csv_dir}")
        self.df = pd.concat(df_list, ignore_index=True)

        # Normalize and prepare fields
        self.df.columns = [c.strip().lower().replace(' ', '_') for c in self.df.columns]
        # Common fields across both CSVs
        # patient_id, breast_density (or breast density), left_or_right_breast, image_view, abnormality_id, pathology, assessment, subtlety
        # calc-specific: calc_type, calc_distribution
        # mass-specific: mass_shape, mass_margins

        # Pre-fit encoders on the whole split
        self.enc_side = OneHotEncoder();            self.enc_side.fit(self.df['left_or_right_breast'])
        self.enc_view = OneHotEncoder();            self.enc_view.fit(self.df['image_view'])
        self.enc_mod  = OneHotEncoder();            self.enc_mod.fit(self.df['__modality__'])
        self.enc_calc_type = OneHotEncoder();       self.enc_calc_type.fit(self.df.get('calc_type', pd.Series(dtype=str)))
        self.enc_calc_dist = OneHotEncoder();       self.enc_calc_dist.fit(self.df.get('calc_distribution', pd.Series(dtype=str)))
        self.enc_mass_shape = OneHotEncoder();      self.enc_mass_shape.fit(self.df.get('mass_shape', pd.Series(dtype=str)))
        self.enc_mass_margins = OneHotEncoder();    self.enc_mass_margins.fit(self.df.get('mass_margins', pd.Series(dtype=str)))

        # Build in-memory index of resolved image paths and tabular/labels
        records = []
        for _, row in self.df.iterrows():
            pid = row['patient_id']
            side = str(row['left_or_right_breast']).upper()
            view = str(row['image_view']).upper()
            abn_id = str(row.get('abnormality_id', ''))
             #print(f"Processing row: {row['patient_id']} {row['left_or_right_breast']} {row['image_view']} {row['__modality__']}")

            modality = row['__modality__']
             #print(f"Modality: {modality}")
            if modality == 'CALC':
                base = 'Calc-Training' if split == 'train' else 'Calc-Test'
            else:
                base = 'Mass-Training' if split == 'train' else 'Mass-Test'
            # Build two possible names: with and without abnormality suffix
            name_base = f"{base}_{pid}_{side}_{view}"
            name_with_abn = f"{name_base}_{abn_id}" if abn_id and abn_id.strip() else name_base

            # Prefer cropped images for training, else fallback
            prefer = ('cropped images', 'full mammogram images', 'ROI mask images') if use_cropped else ('full mammogram images', 'cropped images', 'ROI mask images')
            jpeg_rel_list = self.index.find_jpegs(name_with_abn, prefer=prefer)
            jpeg_rel_list = [p.removeprefix("CBIS-DDSM/") for p in jpeg_rel_list]

            if not jpeg_rel_list:
                jpeg_rel_list = self.index.find_jpegs(name_base, prefer=prefer)
            if not jpeg_rel_list:
                # Skip if no mapping
                continue
            # Use the first available jpeg (paths in dicom_info are like 'CBIS-DDSM/jpeg/...')
            jpeg_rel = jpeg_rel_list[0]
            img_path = jpeg_rel
            if not os.path.isabs(img_path):
                # Make it absolute from repo root (two levels up from csv_dir)
                repo_root = os.path.abspath(os.path.join(self.csv_dir, '..'))
                img_path = os.path.join(repo_root, os.path.normpath(jpeg_rel))

            # Some entries might not exist on disk if dataset is partial
            if not os.path.exists(img_path):
                continue
            # Build tabular features
            breast_density = safe_float(row.get('breast_density', 0))
            assessment = safe_float(row.get('assessment', 0))
            subtlety = safe_float(row.get('subtlety', 0))


            # One-hots
            side_oh = self.enc_side.transform(row.get('left_or_right_breast'))
            view_oh = self.enc_view.transform(row.get('image_view'))
            mod_oh  = self.enc_mod.transform(modality)
            calc_type_oh = self.enc_calc_type.transform(row.get('calc_type'))
            calc_dist_oh = self.enc_calc_dist.transform(row.get('calc_distribution'))
            mass_shape_oh = self.enc_mass_shape.transform(row.get('mass_shape'))
            mass_margins_oh = self.enc_mass_margins.transform(row.get('mass_margins'))

            tab_list = [
                np.array([breast_density, assessment, subtlety], dtype=np.float32),
                side_oh, view_oh, mod_oh,
                calc_type_oh, calc_dist_oh,
                mass_shape_oh, mass_margins_oh,
            ]
            tab = np.concatenate(tab_list, axis=0).astype(np.float32)

            # Label mapping
            pathology = str(row.get('pathology', '')).strip().upper()
            if pathology == 'MALIGNANT':
                label = 1
            else:
                # BENIGN or BENIGN_WITHOUT_CALLBACK -> 0
                label = 0

            records.append({
                'image_path': img_path,
                'tabular': tab,
                'label': label,
            })

        if not records:
            raise RuntimeError('No records could be built from CSVs and dicom_info mapping. Check dataset paths.')

        self.records = records
        # Keep tabular dim for downstream models
        self.tabular_dim = self[0][1].shape[0]
        # Log class distribution for debugging
        label_counts = Counter([r['label'] for r in self.records])
        print(f"CBISDDSMCaseDataset built with {len(self.records)} cases. Class distribution: {dict(label_counts)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Load image as grayscale
        with Image.open(rec['image_path']) as img:
            img = img.convert('L')  # ensure 1 channel
            img_t = self.transform(img)  # [1, H, W]
        # Tabular
        tab_t = torch.from_numpy(rec['tabular'])  # [T]
        y = torch.tensor(rec['label'], dtype=torch.long)
        return img_t, tab_t, y


class MultimodalDataset(Dataset):
    def __init__(self, images, text, tabular, labels, target_depth=128):
        self.images = [resize_volume(img, target_depth) for img in images]
        self.text = text
        self.tabular = tabular
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.text[idx], self.tabular[idx], self.labels[idx]


def train_model(model, data, epochs=10, target_depth=128, val_data=None, batch_size=16, patience=5, use_balanced_sampler=True):
    """
    Train either:
    - on legacy dict data {'images', 'labels', optional 'text','tabular'}
    - or on any torch.utils.data.Dataset yielding (image, tabular, label)

    New:
    - Optional validation Dataset with early stopping.
    - WeightedRandomSampler toggle for class imbalance.
    - Configurable batch size & patience.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'loss': [], 'acc': []}
    if val_data is not None:
        history['val_loss'] = []
        history['val_acc'] = []
    print(f"Entraînement du modèle {model.__class__.__name__} ")

    # Helper to run validation on a Dataset
    def _run_eval(ds: Dataset):
        model.eval()
        loader = DataLoader(ds, batch_size=min(64, batch_size), shuffle=False)
        total_loss, correct, n = 0.0, 0, 0
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    image, text, tabular, y = batch
                else:
                    image, tabular, y = batch
                    text = None
                try:
                    out = model(image, tabular=tabular, text=text) if text is not None else model(image, tabular=tabular)
                except TypeError:
                    out = model(image)
                loss = criterion(out, y)
                total_loss += loss.item() * y.size(0)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                n += y.size(0)
                all_preds.extend(preds.cpu().tolist())
        avg_loss = total_loss / max(1, n)
        acc = correct / max(1, n)
        # Simple collapse detection: count unique predicted classes
        unique_preds = len(set(all_preds))
        if unique_preds < 2:
            print(f"⚠️  Collapse warning: validation predictions cover {unique_preds} class(es).")
        return avg_loss, acc

    # New: handle Dataset directly
    if isinstance(data, Dataset) and not isinstance(data, TensorDataset):
        dataset = data
        # 🔧 Address class imbalance via sampling ONLY (avoid double-counting with loss weights)
        sampler = None
        try:
            if use_balanced_sampler:
                if hasattr(dataset, 'records'):
                    labels = [int(r['label']) for r in dataset.records]
                else:
                    labels = [int(dataset[i][-1]) for i in range(len(dataset))]
                counts = Counter(labels)
                # Inverse-frequency sampling
                sample_weights = torch.tensor([1.0 / counts[l] for l in labels], dtype=torch.double)
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                # Keep unweighted loss when sampler is used
                criterion = nn.CrossEntropyLoss()
                print(f"Using WeightedRandomSampler with inverse-frequency weights; loss is unweighted. Class counts: {dict(counts)}")
            else:
                print("Balanced sampler disabled; using plain shuffling.")
        except Exception as e:
            print(f"Warning: could not set up sampler ({e}); falling back to unweighted loss and shuffle.")

        # 🔧 Optimizer with backbone/head param groups + warmup freeze
        if hasattr(model, 'image_backbone'):
            head_params = [p for n, p in model.named_parameters() if not n.startswith('image_backbone.')]
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('image_backbone.')]
            # Freeze backbone for warmup
            for p in backbone_params:
                p.requires_grad = False
            optimizer = optim.Adam([
                {'params': head_params, 'lr': 1e-3},
                {'params': backbone_params, 'lr': 1e-4},
            ])
            warmup_epochs = max(1, min(3, epochs // 2))
            print(f"Backbone frozen for {warmup_epochs} epoch(s).")
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            warmup_epochs = 0

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True)
        val_loader = None
        if isinstance(val_data, Dataset):
            val_loader = DataLoader(val_data, batch_size=min(64, batch_size), shuffle=False)
        print(f"DataLoader créé avec {len(loader)} batches.")
        print(f"Début de l'entraînement pour {epochs} epochs.")

        best_metric = -float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            # Unfreeze backbone after warmup
            if hasattr(model, 'image_backbone') and epoch == warmup_epochs:
                for p in model.image_backbone.parameters():
                    p.requires_grad = True
                print("Backbone unfrozen.")

            model.train()
            total_loss, correct, seen = 0.0, 0, 0
            loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in loop:
                # Expect (image, tabular, y) or (image, text, tabular, y)
                if len(batch) == 4:
                    image, text, tabular, y = batch
                else:
                    image, tabular, y = batch
                    text = None

                # Try to call model with available modalities; fallback to image-only
                try:
                    if text is not None:
                        out = model(image, tabular=tabular, text=text)
                    else:
                        out = model(image, tabular=tabular)
                except TypeError:
                    out = model(image)

                loss = criterion(out, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                seen += y.size(0)
                loop.set_postfix(loss=loss.item())
            train_acc = correct / max(1, seen)
            train_loss = total_loss / max(1, seen)
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = _run_eval(val_data)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                metric = val_acc
                print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                # Early stopping on best validation accuracy
                if metric > best_metric + 1e-6:
                    best_metric = metric
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}. Best val_acc={best_metric:.4f}")
                        break
            else:
                print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

        # Restore best model if available
        if val_data is not None and best_state is not None:
            model.load_state_dict(best_state)
        return history

    # Legacy paths below
    if 'text' in data and 'tabular' in data:
        # 🧠 Multimodal case
        dataset = MultimodalDataset(data['images'], data['text'], data['tabular'], data['labels'], target_depth)
    else:
        # 🖼️ Image only case
        resized_images = [resize_volume(img, (128, 224, 224)) for img in data['images']]
        inputs = torch.stack(resized_images)
        print("inputs shape:", inputs.shape)
        print("labels shape:", data['labels'])

        dataset = TensorDataset(inputs, data['labels'])

    loader = DataLoader(dataset, batch_size=min(8, batch_size), shuffle=True)
    print(f"DataLoader créé avec {len(loader)} batches.")
    print(f"Début de l'entraînement pour {epochs} epochs.")
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        # Progress bar for batches
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            if 'text' in data:
                image, text, tabular, y = batch
                out = model(image, text, tabular)
            else:
                x, y = batch
                out = model(x)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

            # Optional: update tqdm bar with current loss
            loop.set_postfix(loss=loss.item())

        acc = correct / len(loader.dataset)
        history['loss'].append(total_loss)
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.4f}")

    return history


def train_model_multimodal(model, data, epochs=2, target_depth=128):
    """
    Entraîne un modèle multimodal avec des données d'images, de texte et de tabular.
    
    Args:
        model (nn.Module): Modèle multimodal à entraîner.
        data (dict): Dictionnaire contenant 'images', 'text', 'tabular', 'labels'.
        epochs (int): Nombre d'époques pour l'entraînement.
        target_depth (int): Profondeur cible pour les volumes 3D.
    
    Returns:
        dict: Historique de l'entraînement avec perte et précision.
    """
    print(f"Entraînement du modèle {model.__class__.__name__} pour {epochs} epochs.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'loss': [], 'acc': []}
    dataset = MultimodalDataset(data['images'], data['text'], data['tabular'], data['labels'], target_depth)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"DataLoader créé avec {len(loader)} batches.")
    print(f"Début de l'entraînement pour {epochs} epochs.")
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        # Progress bar for batches
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            image, text, tabular, y = batch
            out = model(image, tabular=tabular, text=text)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

            # Optional: update tqdm bar with current loss
            loop.set_postfix(loss=loss.item())

        acc = correct / len(loader.dataset)
        history['loss'].append(total_loss)
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.4f}")
    return history