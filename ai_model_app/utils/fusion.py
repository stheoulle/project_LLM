import torch.nn as nn
import torch
import numpy as np

class MultimodalFusion(nn.Module):
    def __init__(self, image_backbone, image_dim, text_dim=768, tabular_dim=16):
        super().__init__()
        self.image_backbone = image_backbone
        self.flatten = nn.Flatten()
        self.has_text = text_dim > 0
        self.has_tabular = tabular_dim > 0

        fusion_dim = image_dim + (text_dim if self.has_text else 0) + (tabular_dim if self.has_tabular else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, image, text=None, tabular=None):
        image_feat = self.image_backbone(image)
        image_feat = self.flatten(image_feat)

        features = [image_feat]
        if self.has_text and text is not None:
            features.append(text)
        if self.has_tabular and tabular is not None:
            features.append(tabular)

        x = torch.cat(features, dim=1)
        return self.fusion(x)


def multimodal_fusion_model(image_model, data, patient_idx=0):
    """
    Construit un modèle de fusion multimodal à partir d’un modèle image pré-entraîné,
    en combinant texte et métadonnées si disponibles.

    Args:
        image_model (nn.Module): modèle CNN déjà entraîné
        data (dict): dictionnaire contenant 'images', 'tabular', éventuellement 'text'
        patient_idx (int): index du patient pour extraire les dimensions

    Returns:
        nn.Module: modèle fusion multimodal
    """
    # 🔍 1. Extraire une image pour estimer la dimension de sortie
    image_sample = data["images"][patient_idx]
    if isinstance(image_sample, list):
        image_sample = image_sample[0]
    if not isinstance(image_sample, torch.Tensor):
        image_sample = torch.tensor(image_sample, dtype=torch.float32)

    image_sample = image_sample.unsqueeze(0)  # ajout batch dim : [1, C, D, H, W]
    image_model.eval()
    with torch.no_grad():
        image_output = image_model(image_sample)
        image_output_flat = image_output.view(image_output.size(0), -1)
        image_dim = image_output_flat.shape[1]

    # 📐 2. Calculer les dimensions supplémentaires
    text_dim = 768 if "text" in data and data["text"] is not None else 0
    tabular_sample = data["tabular"][patient_idx]
    tabular_dim = len(tabular_sample) if isinstance(tabular_sample, (list, np.ndarray, torch.Tensor)) else 1

    # 🔀 3. Construire le modèle de fusion
    return MultimodalFusion(
        image_backbone=image_model,
        image_dim=image_dim,
        text_dim=text_dim,
        tabular_dim=tabular_dim
    )
