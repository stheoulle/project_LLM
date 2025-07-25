import torch
import torch.nn as nn
import torchvision.models as models
from monai.networks.nets import ResNet, EfficientNetBN

def load_cnn_model(model_name, input_shape):
    """
    Charge un modèle CNN adapté aux données 2D ou 3D.

    Args:
        model_name (str): 'resnet18', 'efficientnet', 'radimagenet'
        input_shape (tuple): shape de l'entrée, ex: (1, 224, 224) ou (1, 64, 224, 224)

    Returns:
        nn.Module: modèle PyTorch prêt à entraîner
    """
    torch.set_float32_matmul_precision('medium')
    is_3d = len(input_shape) == 4  # [C, D, H, W] pour 3D, [C, H, W] pour 2D
    num_classes = 2  # Modifier si besoin

    if model_name == "resnet18":
        spatial_dims = 3 if len(input_shape) == 4 else 2  # [C, D, H, W] → 3D, sinon 2D
        return ResNet(
            spatial_dims=spatial_dims,
            n_input_channels=input_shape[0],
            num_classes=num_classes,
            block="basic",
            layers=(1,1,1,1),
            block_inplanes=(16, 32, 64, 128)
        )
    elif model_name == "efficientnet":
        if is_3d:
            raise ValueError("EfficientNet B0 3D non disponible. Utilisez ResNet18 pour les IRM 3D.")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # grayscale
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "radimagenet":
        if is_3d:
            raise ValueError("RadImageNet n'est pas prévu pour le 3D.")
        # Exemple générique d’un ResNet50-like. À adapter si tu as un modèle custom.
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # Charger ici un checkpoint si tu en as un (non inclus dans PyTorch)

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return model
