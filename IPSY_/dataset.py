import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pydicom

CLASS_COLUMNS = ["Normal", "Actionable", "Benign", "Cancer"]

def _load_image(path, view, root=None):
    """Charge un volume DICOM 3D et renvoie un numpy array [D, H, W]."""
    full_path = os.path.join(root, path) if root else path
    # Lecture DICOM
    if full_path.endswith(".npy"):
        img = np.load(full_path)
    else:
        # Lecture DICOM, on suppose un dossier avec slices
        # tri par InstanceNumber si nécessaire
        ds = pydicom.dcmread(full_path)
        img = ds.pixel_array.astype(np.float32)
    return img

def resize_tensor(img: torch.Tensor, target_size=(64, 224, 224)) -> torch.Tensor:
    """
    Redimensionne un tenseur 3D [C, D, H, W] ou 4D [D, H, W] vers [C, D_out, H_out, W_out].
    """
    if isinstance(img, np.ndarray):
        img = torch.tensor(img, dtype=torch.float32)  # [D, H, W]

    # ajouter une dimension channel si nécessaire
    if img.ndim == 3:
        img = img.unsqueeze(0)  # [1, D, H, W]

    # ajouter batch dimension pour F.interpolate
    img = img.unsqueeze(0)  # [1, C, D, H, W]
    img = F.interpolate(img, size=target_size, mode='trilinear', align_corners=False)
    img = img.squeeze(0)  # [C, D, H, W]
    return img

class DBTDataset(Dataset):
    """Dataset PyTorch pour classification DBT 3D."""

    def __init__(
        self,
        labels_csv: str,
        filepaths_csv: str,
        root: Optional[str] = None,
        transform=None,
        target_size=(64, 224, 224),
        verbose: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.target_size = target_size
        self.verbose = verbose

        df_labels = pd.read_csv(labels_csv)
        df_paths = pd.read_csv(filepaths_csv)

        # Harmonisation des colonnes
        label_cols = [c for c in CLASS_COLUMNS if c in df_labels.columns]
        if len(label_cols) != 4 and "Actionnable" in df_labels.columns:
            df_labels = df_labels.rename(columns={"Actionnable": "Actionable"})
            label_cols = [c for c in CLASS_COLUMNS if c in df_labels.columns]

        key = ["PatientID", "StudyUID", "View"]
        merged = pd.merge(df_labels, df_paths, on=key)
        merged["label_idx"] = merged[CLASS_COLUMNS].values.argmax(axis=1)
        self.df = merged.reset_index(drop=True)

        # Préparer le pas d'affichage pour environ 5% d'avancement
        total = len(self.df)
        try:
            self._progress_step = max(1, int(total / 20))  # 20 steps => 5% chacun
        except Exception:
            self._progress_step = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_col = "descriptive_path_x"
        path = os.path.join(self.root, row[path_col]) if self.root else row[path_col]

        # Charger le volume 3D
        img = _load_image(path, row["View"], root=self.root)

        # Redimensionner
        img = resize_tensor(img, target_size=self.target_size)

        # Appliquer transformation éventuelle
        if self.transform:
            img = self.transform(img)

        label = row["label_idx"]

        # Affichage d'avancement si demandé
        if self.verbose:
            total = len(self.df)
            # afficher régulièrement tous les _progress_step éléments et au dernier
            if (idx % self._progress_step == 0) or (idx == total - 1):
                pct = (idx + 1) / total * 100 if total > 0 else 100.0
                print(f"Dataset progress: {idx+1}/{total} ({pct:.1f}%)")

        return img, label
