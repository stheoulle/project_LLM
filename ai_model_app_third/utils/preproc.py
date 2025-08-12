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



def prepare_data(modality, mri_types=["MRI"]):
    """Prepare data for different modalities.

    Supported modality values:
      - 'images' (existing behavior)
      - 'images+meta', 'images+meta+reports' (existing behavior)
      - 'docs_only', 'docs_embeddings', 'text' (new): load precomputed text embeddings from docs/ and labels from docs/labels.csv

    Returns a dict with keys depending on modality. For docs/text mode the dict contains:
      {
        'type': 'text',
        'embeddings': torch.Tensor of shape [N, D],
        'labels': torch.LongTensor of shape [N],  (if labels available)
        'paths': list of doc paths (may be empty)
      }

    For image-based modes behavior is unchanged.
    """
    # --- New: handle docs-only mode ---
    if isinstance(modality, str) and modality.lower() in ('docs_only', 'docs_embeddings', 'text', 'text_only'):
        import numpy as _np
        import json
        from pathlib import Path
        import pandas as _pd
        docs_dir = Path('docs')

        # Candidate embedding files (most likely names used by the app)
        candidates = [
            docs_dir / 'embeddings_converted.npz',
            docs_dir / 'embeddings_dense.npz',
            docs_dir / 'embeddings.npz',
            docs_dir / 'embeddings.npz.npz'
        ]

        emb = None
        paths = []
        emb_path_used = None

        for c in candidates:
            if c.exists():
                try:
                    arr = _np.load(str(c), allow_pickle=True)
                    files = getattr(arr, 'files', None)
                    if files and 'embeddings' in files:
                        emb = arr['embeddings']
                        if 'paths' in files:
                            paths = list(arr['paths']) if isinstance(arr['paths'], _np.ndarray) else list(arr['paths'])
                        emb_path_used = str(c)
                        break
                    # If the npz contains sparse saved data or single array, try to interpret
                    if files and len(files) >= 1:
                        first = arr[files[0]]
                        # if first is scipy sparse saved as object or dense, try to use it
                        emb = first
                        if 'paths' in files:
                            paths = list(arr['paths']) if isinstance(arr['paths'], _np.ndarray) else list(arr['paths'])
                        emb_path_used = str(c)
                        break
                except Exception:
                    # not a numpy archive we can read here
                    emb = None
                    continue

        # If not found above, try loading sparse scipy file and its meta json
        if emb is None:
            try:
                from scipy import sparse as _sp
                # try common name
                sparse_candidate = docs_dir / 'embeddings.npz'
                if sparse_candidate.exists():
                    mat = _sp.load_npz(str(sparse_candidate))
                    emb = mat.toarray()
                    meta_json = docs_dir / 'embeddings.npz.meta.json'
                    if meta_json.exists():
                        with open(meta_json, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                            paths = meta.get('paths', []) or []
                    emb_path_used = str(sparse_candidate)
            except Exception:
                emb = None

        if emb is None:
            # Last fallback: scan docs dir for text files and build very simple TF-IDF embeddings in-memory
            txts = []
            doc_paths = []
            for p in docs_dir.glob('**/*'):
                if p.suffix.lower() in ('.txt', '.md'):
                    try:
                        txts.append(p.read_text(encoding='utf-8'))
                        doc_paths.append(str(p))
                    except Exception:
                        continue
            if len(txts) == 0:
                raise RuntimeError(f"No text documents or embeddings found under {docs_dir}")
            # lightweight TF-IDF fallback (in-memory)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer(max_features=20000, stop_words='english')
                mat = vec.fit_transform(txts)
                emb = mat.toarray()
                paths = doc_paths
                emb_path_used = 'in-memory-tfidf'
            except Exception:
                raise RuntimeError("No embeddings found and sklearn not available to build fallback TF-IDF embeddings.")

        # Normalize and convert embeddings to numpy 2D
        emb = _np.asarray(emb)
        # if object array (list of arrays), try to vstack
        if emb.dtype == object:
            try:
                emb = _np.vstack([_np.asarray(x) for x in emb])
            except Exception:
                if emb.size == 1 and isinstance(emb[0], (list, _np.ndarray)):
                    emb = _np.asarray(emb[0])
                else:
                    raise RuntimeError("Embeddings array has unexpected object dtype and cannot be converted to 2D array.")

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Load labels CSV if present
        labels_csv = docs_dir / 'labels.csv'
        labels_arr = None
        if labels_csv.exists():
            try:
                df = _pd.read_csv(labels_csv)
                if 'filename' in df.columns and 'label' in df.columns:
                    # build mapping from basename -> label
                    mapping = {str(r['filename']): r['label'] for _, r in df.iterrows()}
                    # match in order of paths if available, else assume filenames equal order of embeddings
                    if paths and len(paths) == emb.shape[0]:
                        labels_list = []
                        for p in paths:
                            b = os.path.basename(p)
                            lbl = mapping.get(b)
                            labels_list.append(lbl)
                        # If any label is missing, set labels_arr to None so caller can decide
                        if all([l is not None and str(l).strip() != '' for l in labels_list]):
                            # map unique labels to ints
                            uniques = sorted(list(set(labels_list)))
                            label2idx = {lab: idx for idx, lab in enumerate(uniques)}
                            labels_arr = _np.array([label2idx[l] for l in labels_list], dtype=_np.int64)
                    else:
                        # fallback: try to match by filename basenames inside docs directory
                        # build list of basenames from docs files in alphabetical order
                        basenames = [os.path.basename(p) for p in paths] if paths else []
                        if basenames and len(basenames) == emb.shape[0]:
                            labels_list = [mapping.get(b) for b in basenames]
                            if all([l is not None and str(l).strip() != '' for l in labels_list]):
                                uniques = sorted(list(set(labels_list)))
                                label2idx = {lab: idx for idx, lab in enumerate(uniques)}
                                labels_arr = _np.array([label2idx[l] for l in labels_list], dtype=_np.int64)
                # else: CSV exists but wrong format -> ignore
            except Exception:
                labels_arr = None

        # Build return dict (convert embeddings and labels to torch tensors)
        import torch as _torch
        data = {'type': 'text', 'embeddings': _torch.tensor(emb, dtype=_torch.float32)}
        if labels_arr is not None:
            data['labels'] = _torch.tensor(labels_arr, dtype=_torch.long)
        data['paths'] = paths
        data['embeddings_source'] = emb_path_used
        return data

   

    if modality == "images": 
        # --- existing image-based prepare implementation ---
        patient_root = "Breast-diagnosis/manifest-BbshIhaG7188578559074019493/BREAST-DIAGNOSIS"
        patient_ids = os.listdir(patient_root)
        print(f"Nombre de patients trouvés : {len(patient_ids)}")

        data = {'type': None}
        data['images'] = []
        valid_patient_ids = []
        max_patients = 1  # 🔸 Test avec seulement 1 patients valides
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
        # print(metadata)
        print(f"Nombre de patients dans les métadonnées : {len(metadata)}")
        print(f"Nombre de patients valides avec images : {len(valid_patient_ids)}")
        data['tabular'] = []
        print(metadata.keys())
        for pid in valid_patient_ids:
            for key in metadata.keys():
                if not isinstance(key, str):
                    continue  # ignore Nan
                pid_upper = pid.upper()
                key_upper = key.upper()
                if pid_upper == key_upper:
                    print(f"✅ Patient {pid} : métadonnées chargées")
                    # add metadata to data['tabular']
                    tabular_data = metadata[key]
                    if isinstance(tabular_data, dict):
                        data['tabular'].append(list(tabular_data.values()))
                    else:
                        data['tabular'].append(tabular_data)

                    break       
        print(f"Nombre de patients valides avec données tabulaires : {len(data['tabular'])}")

        if len(data['images']) == 0:
            raise RuntimeError("🔴 Aucune image valide chargée pour les patients.")

        data['labels'] = torch.randint(0, 2, (len(data['images']),))  # à adapter plus tard
    return data
