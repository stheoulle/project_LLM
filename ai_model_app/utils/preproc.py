import os
import torch
import pandas as pd
import pydicom
import numpy as np
import torchvision.transforms as T

def load_dicom_as_tensor(path, target_size=(224, 224)):
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)  # Normalisation
        tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]

        # Resize à la taille cible
        resize = T.Resize(target_size)
        tensor = resize(tensor)
        return tensor
    except Exception as e:
        print(f"Erreur lors du chargement de {path} : {e}")
        return None

def load_dicom_images_filtered(patient_dir, mri_types=["MRI"], target_size=(224, 224)):
    tensors = []
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                full_path = os.path.join(root, f)
                if any(mt.upper() in root.upper() for mt in mri_types):
                    tensor = load_dicom_as_tensor(full_path, target_size=target_size)
                    if tensor is not None:
                        tensors.append(tensor)

    if len(tensors) > 0:
        tensors = sorted(tensors, key=lambda x: x.shape[-1])
        volume = torch.stack(tensors)  # [n_slices, 1, H, W]
        volume = volume.permute(1, 0, 2, 3)  # [1, D, H, W] pour 3D CNN si besoin
        return volume
    return None

def load_clinical_metadata(path='Breast-diagnosis/TCIA-Breast-clinical-data-public-7_16_11.xlsx'):
    return pd.read_excel(path)

def prepare_data(modality, mri_types=["MRI"]):
    data = {'type': None}

    patient_root = "Breast-diagnosis/manifest-BbshIhaG7188578559074019493/BREAST-DIAGNOSIS"
    patient_ids = os.listdir(patient_root)
    print(f"Nombre de patients trouvés : {len(patient_ids)}")

    data['images'] = []
    valid_patient_ids = []
    max_patients = 1  # 🔸 Test avec seulement 1 patients valides

    if modality == "images":
        data['type'] = 'image'
        for pid in patient_ids:
            if len(data['images']) >= max_patients:
                break  # 🔸 Arrêter une fois 4 patients valides traités

            patient_dir = os.path.join(patient_root, pid)

            if not os.path.isdir(patient_dir):
                continue

            imgs = load_dicom_images_filtered(patient_dir, mri_types)

            if imgs is not None and isinstance(imgs, torch.Tensor):
                print(f"✅ Patient {pid} : {imgs.shape} images chargées")
                data['images'].append(imgs)
                valid_patient_ids.append(pid)
            else:
                print(f"⚠️ Patient {pid} ignoré (aucune image trouvée ou format incorrect)")

    metadata = load_clinical_metadata()
    data['tabular'] = [metadata[pid] for pid in valid_patient_ids if pid in metadata]

    if len(data['images']) == 0:
        raise RuntimeError("Aucune image valide chargée pour les patients.")

    data['labels'] = torch.randint(0, 2, (len(data['images']),))  # à adapter plus tard
    return data
