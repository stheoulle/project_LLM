import os
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def load_dicom_series(folder_path):
    """Charge un volume DICOM normalisé en float32 entre 0 et 1"""
    slices = []
    for file in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, file)
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(path)
            slices.append(ds.pixel_array)

    volume = np.stack(slices, axis=0)  # (D, H, W)
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

class PatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=(128, 128), stride=128, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        self.patches = []
        self._load_all_patches()

    def _load_all_patches(self):
        for dirpath, _, filenames in os.walk(self.root_dir):
            if any(f.endswith(".dcm") for f in filenames):
                try:
                    volume = load_dicom_series(dirpath)
                    D, H, W = volume.shape
                    for d in range(D):
                        for i in range(0, H - self.patch_size[0] + 1, self.stride):
                            for j in range(0, W - self.patch_size[1] + 1, self.stride):
                                patch = volume[d, i:i+self.patch_size[0], j:j+self.patch_size[1]]
                                self.patches.append((patch, dirpath))  # keep path for label later
                except Exception as e:
                    print(f"Erreur sur {dirpath} : {e}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, path = self.patches[idx]
        image = self.transform(patch)
        
        # ⚠️ Label dummy pour l'instant (ex: 0)
        label = 0  
        return image, label


dataset = PatchDataset("Filtered-CMB-BRCA")
print("Nombre de patches :", len(dataset))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Affichage d'un batch

images, labels = next(iter(dataloader))
plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i][0], cmap='gray')  # image[i][0] car image est (1, H, W)
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
