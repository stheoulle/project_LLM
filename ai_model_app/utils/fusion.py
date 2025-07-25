import torch.nn as nn
import torch
from models.cnn_backbones import load_cnn_model

class MultimodalFusion(nn.Module):
    def __init__(self, image_dim, text_dim=768, tabular_dim=16):
        super().__init__()
        self.transformer = nn.Sequential(
            nn.Linear(image_dim + text_dim + tabular_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, image_feat, text_feat, tabular_feat):
        x = torch.cat([image_feat, text_feat, tabular_feat], dim=1)
        return self.transformer(x)

def multimodal_fusion_model(model_choice, data):
    image_backbone = load_cnn_model(model_choice)
    return nn.Sequential(image_backbone, MultimodalFusion(512))
