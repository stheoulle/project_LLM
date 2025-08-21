"""Backbone selection and initialization utilities
Supports: ResNet-50/101, EfficientNet-B4, ConvNeXt-T/B via timm.
Provides helpers to adapt first conv for single-channel mammography and to load RadImageNet or ImageNet weights.

Usage:
    from model_backbones import get_backbone
    model, feat_dim = get_backbone('resnet50', pretrained=True, in_channels=1, radimagenet_path=None)

Dependencies: torch, timm
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _adapt_first_conv(model: nn.Module, in_channels: int = 1):
    """Adapt the model first conv layer to accept `in_channels` instead of 3.
    Attempts common attribute names used by timm/backbones and torchvision.
    It copies pretrained weights by averaging across the input channels.
    """
    conv_names = [
        'conv1',               # torchvision resnets
        'stem.conv',           # some timm models
        'patch_embed.proj',    # vit/convnext-like
        'features.0',
        'stem',
    ]

    for name in conv_names:
        parts = name.split('.')
        m = model
        parent = None
        attr = None
        try:
            for p in parts:
                parent = m
                m = getattr(m, p)
                attr = p
        except Exception:
            continue
        if isinstance(m, nn.Conv2d):
            old_conv = m
            if old_conv.in_channels == in_channels:
                return model
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
            # initialize new conv
            with torch.no_grad():
                if old_conv.weight is not None:
                    # average weights across channel dimension
                    w = old_conv.weight.data
                    if w.shape[1] == 3:
                        w_mean = w.mean(dim=1, keepdim=True)
                        new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1) / in_channels * 3.0
                    else:
                        # generic: repeat or truncate
                        c_old = w.shape[1]
                        if in_channels < c_old:
                            new_conv.weight.data = w[:, :in_channels, :, :].clone()
                        else:
                            # repeat channels
                            reps = int((in_channels + c_old - 1) / c_old)
                            new_w = w.repeat(1, reps, 1, 1)[:, :in_channels, :, :].clone()
                            new_conv.weight.data = new_w
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data.clone()
            # attach
            setattr(parent, attr, new_conv)
            return model
    # if nothing matched, return model unchanged
    return model


def get_backbone(name: str = 'resnet50', pretrained: bool = True, in_channels: int = 1, radimagenet_path: Optional[str] = None) -> Tuple[nn.Module, int]:
    """Return a backbone model (feature extractor) and number of output features.

    name: timm model name or friendly name: 'resnet50', 'resnet101', 'efficientnet_b4', 'convnext_tiny', 'convnext_base'
    pretrained: if True, use ImageNet pretrained weights from timm unless radimagenet_path provided
    in_channels: 1 for mammography (grayscale)
    radimagenet_path: optional path to a checkpoint (.pth) with RadImageNet weights. If provided, they will be loaded with strict=False.

    Returns (model, feat_dim)
    """
    try:
        import timm
    except Exception as e:
        raise RuntimeError('timm is required for backbone creation. Install with: pip install timm') from e

    # map friendly names to timm model ids
    name_map = {
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'efficientnet_b4': 'tf_efficientnet_b4_ns',
        'convnext_tiny': 'convnext_tiny',
        'convnext_base': 'convnext_base',
    }
    model_name = name_map.get(name, name)

    # create model with no classifier head
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')

    # adapt first conv for grayscale
    if in_channels != 3:
        model = _adapt_first_conv(model, in_channels=in_channels)

    # load RadImageNet checkpoint if provided
    if radimagenet_path is not None:
        try:
            state = torch.load(radimagenet_path, map_location='cpu')
            sd = state.get('state_dict', state)
            # try to remove prefix if present
            new_sd = {}
            for k, v in sd.items():
                k2 = k
                if k.startswith('module.'):
                    k2 = k[len('module.'):]
                new_sd[k2] = v
            model.load_state_dict(new_sd, strict=False)
        except Exception as e:
            # don't crash, warn user
            print(f'Warning: failed to load RadImageNet weights from {radimagenet_path}: {e}')

    # infer feature dim
    feat_dim = getattr(model, 'num_features', None)
    if feat_dim is None:
        # attempt forward on dummy to get dim
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            out = model.forward_features(dummy) if hasattr(model, 'forward_features') else model(dummy)
            if hasattr(out, 'shape'):
                feat_dim = out.shape[1]
            else:
                feat_dim = 512

    return model, int(feat_dim)


if __name__ == '__main__':
    # simple test
    m, d = get_backbone('resnet50', pretrained=True, in_channels=1)
    print(m)
    print('feat_dim=', d)
