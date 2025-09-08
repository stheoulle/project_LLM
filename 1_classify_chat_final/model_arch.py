"""Global-Local architecture for mammography (GMIC/GLAM style) with MIL-style patch aggregation.

Provides:
- GlobalLocalModel: takes a global full-image tensor and optional patches and returns logits.

Forward signatures:
    global_imgs: Tensor[B, C, H, W]
    patches: Tensor[T, C, Ph, Pw] or None  (T = total patches across batch)
    patch_counts: List[int] of length B describing how many patches belong to each sample (sums to T)

The model uses timm backbones via get_backbone (model_backbones.get_backbone).

Design notes:
- Global backbone produces a single feature vector per image (global context).
- Local backbone encodes patches (shared weights), aggregated per case using attention pooling (MIL attention).
- Fusion head concatenates global+local and predicts logits.

This is a minimal, extensible implementation useful for prototyping.
"""
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_backbones import get_backbone


class AttentionPool(nn.Module):
    """Simple attention pooling over variable-length instance sets.
    Computes attention weights for each instance and returns weighted sum per-group.
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats: torch.Tensor, counts: Optional[List[int]] = None):
        """feats: Tensor[T, D] concatenated instances across batch
        counts: List[int] with length B giving number of instances per sample, or None (assume 1 per sample)
        Returns: Tensor[B, D] aggregated features
        """
        if counts is None:
            # assume each sample has exactly one instance
            return feats
        # compute attention logits
        scores = self.attn(feats).squeeze(-1)  # (T,)
        outputs = []
        idx = 0
        for c in counts:
            if c == 0:
                # no patches for this sample -> zero vector
                outputs.append(torch.zeros(feats.shape[1], device=feats.device, dtype=feats.dtype))
                continue
            block = feats[idx: idx + c]  # (c, D)
            block_scores = scores[idx: idx + c]
            alpha = F.softmax(block_scores, dim=0).unsqueeze(-1)  # (c,1)
            agg = (alpha * block).sum(dim=0)  # (D,)
            outputs.append(agg)
            idx += c
        out = torch.stack(outputs, dim=0)  # (B, D)
        return out


class GlobalLocalModel(nn.Module):
    def __init__(self,
                 global_backbone: str = 'resnet50',
                 local_backbone: Optional[str] = None,
                 pretrained: bool = True,
                 in_channels: int = 1,
                 radimagenet_path: Optional[str] = None,
                 attn_hidden: int = 128,
                 fusion_hidden: int = 512,
                 num_classes: int = 1,
                 share_local_global: bool = True):
        super().__init__()
        # create global backbone
        self.global_model, self.global_dim = get_backbone(global_backbone, pretrained=pretrained, in_channels=in_channels, radimagenet_path=radimagenet_path)

        # local backbone: either same as global (shared) or a separate one
        lb_name = local_backbone or global_backbone
        if share_local_global:
            self.local_model = self.global_model
            self.local_dim = self.global_dim
            self._shared_backbone = True
        else:
            self.local_model, self.local_dim = get_backbone(lb_name, pretrained=pretrained, in_channels=in_channels, radimagenet_path=radimagenet_path)
            self._shared_backbone = False

        # attention pooling over patch features
        self.attn_pool = AttentionPool(self.local_dim, hidden_dim=attn_hidden)

        # fusion head
        fused_dim = self.global_dim + self.local_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes)
        )

    def _encode_global(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, C, H, W)
        # many timm models provide forward_features; otherwise call model to get pooled features
        if hasattr(self.global_model, 'forward_features'):
            feats = self.global_model.forward_features(imgs)
        else:
            feats = self.global_model(imgs)
        # if features are spatial (B, D, 1,1) or (B,D,H,W) reduce to vector
        if feats.dim() > 2:
            feats = feats.view(feats.shape[0], -1)
        return feats  # (B, D)

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (T, C, Ph, Pw) return (T, D)
        if patches is None:
            return None
        if hasattr(self.local_model, 'forward_features'):
            feats = self.local_model.forward_features(patches)
        else:
            feats = self.local_model(patches)
        if feats.dim() > 2:
            feats = feats.view(feats.shape[0], -1)
        return feats

    def forward(self, global_imgs: torch.Tensor, patches: Optional[torch.Tensor] = None, patch_counts: Optional[List[int]] = None):
        """Forward pass.
        global_imgs: (B, C, H, W)
        patches: (T, C, Ph, Pw) or None
        patch_counts: list of length B summing to T

        Returns logits Tensor[B, num_classes]
        """
        B = global_imgs.shape[0]
        g_feats = self._encode_global(global_imgs)  # (B, Dg)

        if patches is None or patch_counts is None:
            # no local information: set local aggregation to zeros
            device = g_feats.device
            l_agg = torch.zeros((B, self.local_dim), device=device, dtype=g_feats.dtype)
        else:
            p_feats = self._encode_patches(patches)  # (T, Dl)
            l_agg = self.attn_pool(p_feats, counts=patch_counts)  # (B, Dl)

        fused = torch.cat([g_feats, l_agg], dim=1)  # (B, Dg+Dl)
        logits = self.fusion(fused)
        return logits


if __name__ == '__main__':
    # simple smoke test with dummy inputs
    model = GlobalLocalModel(global_backbone='resnet50', pretrained=False, in_channels=1, share_local_global=True)
    g = torch.randn(2, 1, 1024, 1024)
    # create 6 patches total: sample1 -> 4 patches, sample2 -> 2 patches
    p = torch.randn(6, 1, 256, 256)
    out = model(g, patches=p, patch_counts=[4, 2])
    print('out', out.shape)
