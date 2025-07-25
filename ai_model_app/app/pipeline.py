from utils.preproc import prepare_data
from utils.fusion import multimodal_fusion_model
from utils.metrics import evaluate_model, plot_metrics
from models.cnn_backbones import load_cnn_model
from models.trained import train_model
import torch

def define_steps(modality_choice, mri_types=None):
    data = prepare_data(modality_choice, mri_types)
    return data

def initialize_model(model_choice, data):
    if data['type'] == 'image':
        print(f"🧪 data['images'] contient {len(data['images'])} éléments")

        if len(data['images']) == 0:
            raise ValueError("Aucune image n'a été trouvée dans les données.")

        sample_image = data['images'][0]

        if isinstance(sample_image, list):
            if len(sample_image) == 0:
                raise ValueError("Le premier élément de data['images'] est une liste vide.")
            sample_image = sample_image[0]

        if not isinstance(sample_image, torch.Tensor):
            sample_image = torch.tensor(sample_image)

        input_shape = sample_image.shape
        print(f"📐 input_shape détectée : {input_shape}")
        return load_cnn_model(model_choice, input_shape)
    else:
        return multimodal_fusion_model(model_choice, data)

def run_pipeline(modality_choice, model_choice, mri_types=None):
    print(f"Modèle choisi : {model_choice}")
    print(f"Modalités choisies : {modality_choice}")
    if mri_types:
        print(f"Types de MRI sélectionnés : {mri_types}")

    data = define_steps(modality_choice, mri_types)
    model = initialize_model(model_choice, data)
    
    print("\n🚀 Entraînement du modèle...")
    history = train_model(model, data)
    
    print("\n📈 Évaluation du modèle...")
    results = evaluate_model(model, data)
    
    # if ui:
    #     return history, results
    # else:
    plot_metrics(history, results)

