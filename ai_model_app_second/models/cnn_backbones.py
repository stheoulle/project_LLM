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
    print(f"Chargement du modèle {model_name} avec input_shape {input_shape}")

    if model_name == "resnet18":
        if is_3d:
            # Utiliser MONAI ResNet pour le 3D
            model = ResNet(
                spatial_dims=3,
                n_input_channels=input_shape[0],
                num_classes=num_classes,
                block="basic",
                layers=(2, 2, 2, 2),
                block_inplanes=(64, 128, 256, 512),
            )
            # Garder la tête de classification pour l'entraînement image-only
            return model
        else:
            # Utiliser torchvision ResNet18 pour le 2D
            tv_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Adapter la première couche pour l'échelle de gris
            tv_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Adapter la tête de classification
            tv_model.fc = nn.Linear(tv_model.fc.in_features, num_classes)
            print(f"Modèle resnet18 2D chargé avec succès.")
            return tv_model

    elif model_name == "efficientnet":
        if is_3d:
            raise ValueError("EfficientNet B0 3D non disponible. Utilisez ResNet18 pour les IRM 3D.")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # grayscale
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        print(f"Modèle efficientnet 2D chargé avec succès.")
        return model

    elif model_name == "radimagenet":
        if is_3d:
            raise ValueError("RadImageNet n'est pas prévu pour le 3D.")
        # Exemple générique d’un ResNet50-like. À adapter si tu as un modèle custom.
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"Modèle radimagenet (resnet50) 2D chargé avec succès.")
        return model

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
