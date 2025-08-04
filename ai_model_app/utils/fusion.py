import torch.nn as nn
import torch
import numpy as np

import torch
import torch.nn as nn

class MultimodalTransformerFusion(nn.Module):
    def __init__(self, image_backbone, image_dim=512, tabular_dim=16, text_dim=0, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.image_backbone = image_backbone
        self.has_text = text_dim > 0
        self.has_tabular = tabular_dim > 0

        # Project modalities to same hidden dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        if self.has_tabular:
            self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)
        if self.has_text:
            self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Positional embedding for modality tokens (image, tabular, text)
        self.positional_encoding = nn.Parameter(torch.randn(1, 3, hidden_dim))  # up to 3 modalities

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # binary classification
        )

    def forward(self, image, tabular=None, text=None):
        image_feat = self.image_backbone(image)
        image_feat = image_feat.view(image_feat.size(0), -1)  # flatten
        tokens = [self.image_proj(image_feat)]

        if self.has_tabular and tabular is not None:
            tokens.append(self.tabular_proj(tabular))
        if self.has_text and text is not None:
            tokens.append(self.text_proj(text))

        # Stack into a "sequence" of tokens
        x = torch.stack(tokens, dim=1)  # shape: [batch_size, num_modalities, hidden_dim]
        x = x + self.positional_encoding[:, :x.size(1), :]  # add positional encoding

        x = self.transformer_encoder(x)  # shape: [batch_size, num_modalities, hidden_dim]

        # Pooling: use first token (e.g., image) or mean pooling
        pooled = x.mean(dim=1)  # [batch_size, hidden_dim]

        return self.classifier(pooled)


def multimodal_fusion_model(image_model, data, patient_idx=0, hidden_dim=256, n_heads=4, n_layers=2):
    """
    Construit un modèle de fusion multimodal avec Transformers à partir d’un modèle image pré-entraîné,
    en combinant texte et métadonnées si disponibles.

    Args:
        image_model (nn.Module): modèle CNN déjà entraîné
        data (dict): dictionnaire contenant 'images', 'tabular', éventuellement 'text'
        patient_idx (int): index du patient pour estimer les dimensions
        hidden_dim (int): dimension cachée pour la fusion transformer
        n_heads (int): nombre de têtes pour le multi-head attention
        n_layers (int): nombre de couches dans l'encodeur transformer

    Returns:
        nn.Module: modèle de fusion multimodal avec transformer
    """


    # 🔍 1. Extraire une image pour estimer la dimension de sortie
    image_sample = data["images"][patient_idx]
    if isinstance(image_sample, list):
        image_sample = image_sample[0]
    if not isinstance(image_sample, torch.Tensor):
        image_sample = torch.tensor(image_sample, dtype=torch.float32)

    image_sample = image_sample.unsqueeze(0)  # ajout batch dim : [1, C, D, H, W] ou [1, C, H, W]
    image_model.eval()
    with torch.no_grad():
        image_output = image_model(image_sample)
        image_output_flat = image_output.view(image_output.size(0), -1)
        image_dim = image_output_flat.shape[1]

    # 📐 2. Calculer les dimensions supplémentaires
    text_dim = 768 if "text" in data and data["text"] is not None else 0
    tabular_sample = data["tabular"][patient_idx]
    tabular_dim = len(tabular_sample) if isinstance(tabular_sample, (list, np.ndarray, torch.Tensor)) else 1

    # 🔀 3. Construire le modèle de fusion avec Transformer
    return MultimodalTransformerFusion(
        image_backbone=image_model,
        image_dim=image_dim,
        tabular_dim=tabular_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers
    )
