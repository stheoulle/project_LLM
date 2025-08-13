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

import pandas as pd

def load_clinical_metadata(file_path='Breast-diagnosis/TCIA-Breast-clinical-data-public-7_16_11.xlsx'):
    try:
        df = pd.read_excel(file_path)
        metadata = {}
        for _, row in df.iterrows():
            patient_id = row['Breast Dx Case']
            metadata[patient_id] = row.to_dict()
        return metadata
    except Exception as e:
        print(f"Erreur lors du chargement des métadonnées : {e}")
        return {}


import os
import torch
import math
from collections import Counter

def prepare_data(modality, mri_types=["MRI"]):
    data = {'type': None}

    patient_root = "Breast-diagnosis/manifest-BbshIhaG7188578559074019493/BREAST-DIAGNOSIS"
    patient_ids = os.listdir(patient_root)
    print(f"Nombre de patients trouvés : {len(patient_ids)}")

    data['images'] = []
    valid_patient_ids = []
    max_patients = 89  # for testing

    if modality == "images":
        data['type'] = 'image'
        for pid in patient_ids:
            if len(data['images']) >= max_patients:
                break

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
    print(f"Nombre de patients dans les métadonnées : {len(metadata)}")
    print(f"Nombre de patients valides avec images : {len(valid_patient_ids)}")

    data['tabular'] = []
    raw_labels = []

    for pid in valid_patient_ids:
        matched = False
        for key in metadata.keys():
            if not isinstance(key, str):
                continue  # ignore NaN keys
            if pid.upper() == key.upper():
                print(f"✅ Patient {pid} : métadonnées chargées")
                tabular_data = metadata[key]
                if isinstance(tabular_data, dict):
                    data['tabular'].append(list(tabular_data.values()))
                    label = tabular_data.get("Pathology", None)
                    raw_labels.append(label)
                    print(f"🔍 Label brut : {label}")
                else:
                    data['tabular'].append(tabular_data)
                    raw_labels.append(None)
                    print("⚠️ label is None")
                matched = True
                break
        if not matched:
            raw_labels.append(None)

    print(f"Nombre de patients valides avec données tabulaires : {len(data['tabular'])}")

    if len(data['images']) == 0:
        raise RuntimeError("🔴 Aucune image valide chargée pour les patients.")

    # === 🔧 Process and normalize labels ===
    normalized_labels = []
    final_images = []
    final_tabular = []

    for img, tab, lbl in zip(data['images'], data['tabular'], raw_labels):
        if lbl is None or (isinstance(lbl, float) and math.isnan(lbl)):
            continue  # skip missing labels

        # Normalize integer labels if 0 is used for 'Benign'
        if lbl == 0:
            lbl = "Benign"

        if isinstance(lbl, str) or isinstance(lbl, int):
            norm_label = str(lbl).strip().lower()
            normalized_labels.append(norm_label)
            final_images.append(img)
            final_tabular.append(tab)

    if not normalized_labels:
        raise RuntimeError("🔴 Aucun label utilisable après nettoyage.")

    # === 🔢 Encode labels ===
    unique_labels = sorted(set(normalized_labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"🔁 Encodage des labels : {label_to_int}")

    encoded_labels = [label_to_int[lbl] for lbl in normalized_labels]

    # === ✅ Final data assignment ===
    data['images'] = final_images
    data['tabular'] = final_tabular
    data['labels'] = torch.tensor(encoded_labels)

    # === 📊 Print class distribution ===
    class_counts = Counter(encoded_labels)
    print("\n📊 Répartition des classes :")
    for class_id, count in class_counts.items():
        label_name = [k for k, v in label_to_int.items() if v == class_id][0]
        print(f"  Classe '{label_name}' (id={class_id}): {count} patients")

    print(f"\n✅ Nombre total de labels : {len(data['labels'])}")
    print(f"📐 Images : {len(data['images'])}, Labels : {data['labels'].shape}")

    return data
