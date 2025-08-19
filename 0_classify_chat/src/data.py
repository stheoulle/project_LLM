import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

# Add pydicom import optional
try:
    import pydicom
except Exception:
    pydicom = None


def load_image_generic(path):
    """Load image from path. Supports common image formats and DICOM (.dcm).
    Returns HxWxC uint8 RGB numpy array."""
    p = Path(path)
    if p.suffix.lower() == '.dcm':
        if pydicom is None:
            raise RuntimeError('pydicom is required to read DICOM files. Please install pydicom.')
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array
        # Handle Photometric Interpretation MONOCHROME1/2 inversion if needed
        try:
            photometric = getattr(ds, 'PhotometricInterpretation', '').upper()
            if 'MONOCHROME1' in photometric:
                arr = np.max(arr) - arr
        except Exception:
            pass
        # normalize to 0-255
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype('uint8')
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr]*3, axis=-1)
        return arr
    else:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f'Unable to read image: {p}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class ImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, image_col='image_path', label_col='label'):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.root = Path(root_dir)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = str(row[self.image_col])
        img_path = (self.root / img_rel).resolve()
        img = load_image_generic(img_path)
        if self.transform:
            img = self.transform(img)
        else:
            # ensure tensor
            from torchvision.transforms import ToTensor
            img = ToTensor()(img)
        label = int(row[self.label_col])
        return img, label


def get_dataloader(csv_path, root_dir, batch_size=16, num_workers=4, shuffle=True):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = ImageDataset(csv_path, root_dir, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl
