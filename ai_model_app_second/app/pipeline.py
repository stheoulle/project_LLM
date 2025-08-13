from utils.preproc import prepare_data
from utils.fusion import multimodal_fusion_model
from utils.metrics import evaluate_model, plot_metrics
from models.cnn_backbones import load_cnn_model
from models.trained import train_model
import torch
import os
from torch.utils.data import Dataset

# New import: CBIS-DDSM case-level dataset
from models.trained import CBISDDSMCaseDataset

# New imports for splitting
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def define_steps(modality_choice, mri_types=None):
    # New: special modality to use CBIS-DDSM case-level dataset
    if modality_choice == 'cbis_cases':
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(project_root, 'CBIS-DDSM', 'cbis-ddsm-breast-cancer-image-dataset')
        if not os.path.isdir(base_dir):
            raise RuntimeError(f"CBIS-DDSM base_dir not found: {base_dir}")
        dataset = CBISDDSMCaseDataset(base_dir=base_dir, split='train', use_cropped=True, img_size=224)
        return dataset
    # Legacy
    data = prepare_data(modality_choice, mri_types)
    return data


def get_input_shape_from_sample(image):
    if isinstance(image, list):
        if len(image) == 0:
            raise ValueError("Le premier élément de data['images'] est une liste vide.")
        image = image[0]
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    return image.shape


def initialize_image_model(model_choice, data):
    # Dataset path
    if isinstance(data, Dataset):
        # Expect dataset[i] -> (image, tabular, label) or (image, text, tabular, label)
        sample = data[0]
        image = sample[0]
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        input_shape = image.shape  # [C, H, W]
        return load_cnn_model(model_choice, input_shape)

    # Legacy dict path
    if 'images' not in data or len(data['images']) == 0:
        raise ValueError("Aucune image n'a été trouvée dans les données.")
    print(f"🧪 data['images'] contient {len(data['images'])} éléments")
    input_shape = get_input_shape_from_sample(data['images'][0])
    print(f"📐 input_shape détectée : {input_shape}")
    return load_cnn_model(model_choice, input_shape)


def run_pipeline(
    modality_choice,
    model_choice,
    mri_types=None,
    ui=False,
    # Training params with sensible defaults
    epochs_img=10,
    epochs_fusion=10,
    batch_size=32,
    patience_img=3,
    patience_fusion=5,
    use_balanced_sampler=True,
    val_size=0.2,
    random_state=42,
):
    print(f"Modèle choisi : {model_choice}")
    print(f"Modalités choisies : {modality_choice}")
    if mri_types:
        print(f"Types de MRI sélectionnés : {mri_types}")

    # 🧩 Préparation des données
    data = define_steps(modality_choice, mri_types)

    # If Dataset, create a stratified train/val split
    val_data = None
    train_data = data
    if isinstance(data, Dataset):
        labels = [int((data[i][-1]).item() if hasattr(data[i][-1], 'item') else data[i][-1]) for i in range(len(data))]
        idxs = np.arange(len(data))
        if len(set(labels)) > 1 and len(data) >= 5:
            train_idx, val_idx = train_test_split(
                idxs, test_size=val_size, random_state=random_state, stratify=labels
            )
        else:
            # Fallback non-stratified split if only one class or too small
            test_size = max(1, int(val_size * len(data)))
            train_idx, val_idx = train_test_split(
                idxs, test_size=test_size, random_state=random_state
            )
        train_data = Subset(data, train_idx.tolist())
        val_data = Subset(data, val_idx.tolist())
        print(f"Dataset split: train={len(train_data)}, val={len(val_data)}")

    # Phase 1 — entraînement du modèle image only
    image_model = initialize_image_model(model_choice, train_data)
    print(f"Début de la phase 1 avec le modèle {model_choice} et les données préparées.")
    history = train_model(
        image_model,
        train_data,
        val_data=val_data,
        patience=patience_img,
        batch_size=batch_size,
        use_balanced_sampler=use_balanced_sampler,
        epochs=epochs_img,
    )
    print("Phase 1 terminée : modèle image entraîné.")
    print("Évaluation du modèle image...")
    eval_dataset = val_data if isinstance(train_data, Dataset) else data
    results = evaluate_model(image_model, eval_dataset)
    print("Phase 1 terminée : modèle image entraîné et évalué.")

    # Phase 2 — entraînement du modèle multimodal (avec le modèle image déjà entraîné)
    print(f"Début de la phase 2 avec le modèle {model_choice} et les données préparées.")
    fusion_model = multimodal_fusion_model(image_model, train_data)
    print(f"Modèle de fusion multimodal créé avec {model_choice} comme backbone.")
    fusion_history = train_model(
        fusion_model,
        train_data,
        val_data=val_data,
        patience=patience_fusion,
        batch_size=batch_size,
        use_balanced_sampler=use_balanced_sampler,
        epochs=epochs_fusion,
    )
    print("Phase 2 terminée : modèle multimodal entraîné.")
    fusion_results = evaluate_model(fusion_model, eval_dataset)
    print("Évaluation du modèle multimodal terminée.")

    # Only plot when not called from Streamlit (avoid blocking UI)
    if not ui:
        plot_metrics(fusion_history, fusion_results)
        print("Pipeline complet : modèle multimodal entraîné et évalué.")

    return fusion_history, fusion_results

