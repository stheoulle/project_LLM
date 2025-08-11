from utils.preproc import prepare_data
from utils.fusion import multimodal_fusion_model
from utils.metrics import evaluate_model, plot_metrics
from models.cnn_backbones import load_cnn_model
from models.trained import train_model
import torch

def define_steps(modality_choice, mri_types=None):
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
    if 'images' not in data or len(data['images']) == 0:
        raise ValueError("Aucune image n'a été trouvée dans les données.")
    print(f"🧪 data['images'] contient {len(data['images'])} éléments")
    input_shape = get_input_shape_from_sample(data['images'][0])
    print(f"📐 input_shape détectée : {input_shape}")
    return load_cnn_model(model_choice, input_shape)

def run_pipeline(modality_choice, model_choice, mri_types=None, llm_context=None, extra_inputs=None):
    """
    Run the training pipeline. Accepts optional llm_context (assistant suggestion text or dict)
    and extra_inputs (e.g., uploaded files). These are currently logged and returned but not
    used directly by the training routines. Backward compatible with previous signature.
    """
    print(f"Modèle choisi : {model_choice}")
    print(f"Modalités choisies : {modality_choice}")
    if mri_types:
        print(f"Types de MRI sélectionnés : {mri_types}")
    if llm_context:
        print(f"LLM context / suggestion: {llm_context}")
    if extra_inputs:
        print(f"Extra inputs provided: {list(extra_inputs.keys()) if isinstance(extra_inputs, dict) else type(extra_inputs)}")

    # 🧩 Préparation des données
    data = define_steps(modality_choice, mri_types)

    # Phase 1 — entraînement du modèle image only
    image_model = initialize_image_model(model_choice, data)
    history = train_model(image_model, data)
    results = evaluate_model(image_model, data)
    try:
        plot_metrics(history, results)
    except Exception:
        print("plot_metrics failed or running headless; continuing.")
    print("Phase 1 terminée : modèle image entraîné et évalué.")
    print(f"Début de la phase 2 avec le modèle {model_choice} et les données préparées.")

    # Phase 2 — entraînement du modèle multimodal (avec le modèle image déjà entraîné)
    fusion_model = multimodal_fusion_model(image_model, data)
    print(f"Modèle de fusion multimodal créé avec {model_choice} comme backbone.")
    fusion_history = train_model(fusion_model, data)
    print("Phase 2 terminée : modèle multimodal entraîné.")
    fusion_results = evaluate_model(fusion_model, data)
    print("Évaluation du modèle multimodal terminée.")
    try:
        plot_metrics(fusion_history, fusion_results)
    except Exception:
        print("plot_metrics failed for fusion model; continuing.")
    print("Pipeline complet : modèle multimodal entraîné et évalué.")

    # Return a summary dictionary so callers (UI/CLI) can display or log info
    return {
        'phase1_history': history,
        'phase1_results': results,
        'fusion_history': fusion_history,
        'fusion_results': fusion_results,
        'llm_context': llm_context,
        'extra_inputs': extra_inputs
    }

