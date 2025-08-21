import os
from typing import Optional, List

import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
import torch

from duke_dbt_data import dcmread_image

import torch.nn.functional as F


CLASS_COLUMNS = ["Normal", "Actionable", "Benign", "Cancer"]


def _load_image(path: str, view: str, root: Optional[str] = None) -> np.ndarray:
    if root is not None:
        path = os.path.join(root, path)
    # path may be a relative path inside the dataset
    arr = dcmread_image(path, view)
    # ensure float32
    img = arr.astype(np.float32)
    # rescale intensities to 0..1
    img = rescale_intensity(img, in_range='image', out_range=(0.0, 1.0))
    img = torch.from_numpy(img).float()  # conversion en tensor float
    return img


def resize_tensor(img, target_size=(85, 224, 224)):
    """
    Redimensionne un tensor 3D [D, H, W] vers target_size.
    """
    img = img.unsqueeze(0)  # ajoute batch dim: [1, D, H, W]
    img = F.interpolate(img.unsqueeze(0).float(), size=target_size, mode='trilinear', align_corners=False)
    img = img.squeeze(0).squeeze(0)
    return img

class DBTDataset(Dataset):
    """PyTorch Dataset pour classification IPSY DBT avec redimensionnement automatique."""

    def __init__(
        self,
        labels_csv: str,
        filepaths_csv: str,
        root: Optional[str] = None,
        transform=None,
        target_size=(85, 224, 224),  # profondeur, hauteur, largeur
    ):
        self.root = root
        self.transform = transform
        self.target_size = target_size

        df_labels = pd.read_csv(labels_csv)
        df_paths = pd.read_csv(filepaths_csv)

        # Harmoniser noms de colonnes
        label_cols = [c for c in CLASS_COLUMNS if c in df_labels.columns]
        if len(label_cols) != 4:
            if "Actionnable" in df_labels.columns:
                df_labels = df_labels.rename(columns={"Actionnable": "Actionable"})
            label_cols = [c for c in CLASS_COLUMNS if c in df_labels.columns]

        key = ["PatientID", "StudyUID", "View"]
        merged = pd.merge(df_labels, df_paths, on=key)

        # Calculer index de classe
        merged["label_idx"] = merged[["Normal", "Actionable", "Benign", "Cancer"]].values.argmax(axis=1)

        self.df = merged.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_col = "descriptive_path_x"
        path = os.path.join(self.root, row[path_col])
        img = _load_image(path, row["View"], root=self.root)  # numpy array [D, H, W]

        # convertir en tensor float et ajouter channel dim
        img = torch.tensor(img, dtype=torch.float32)          # [D, H, W]
        img = img.unsqueeze(0)                                # [1, D, H, W] -> 1 channel
        img = img.unsqueeze(0)                                # [1, 1, D, H, W] -> batch dim simulée pour interpolate

        # resize vers target_size
        target_d = min(img.shape[2], 64)  # optionnel : limiter le nombre de slices
        img = F.interpolate(img, size=(target_d, *self.target_size), mode='trilinear', align_corners=False)
        
        img = img.squeeze(0)  # retirer batch dim simulée -> [1, D, H, W]

        if self.transform:
            img = self.transform(img)

        label = row["label_idx"]
        return img, label
