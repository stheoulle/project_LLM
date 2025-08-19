Practical recipe for breast DICOM training (mammography)

Summary

- Goal: Build a robust mammography classifier with global + local pathways and multi-view fusion.
- Keep high-fidelity images (12–16 bit) and preserve physical spacing.

1) Preprocessing (must)

- Preserve bit depth: load raw DICOM pixel arrays, avoid 8-bit conversion. Use dtype float32 for processing but keep original dynamic range. Apply rescale slope/intercept (RescaleSlope/RescaleIntercept) before any intensity transforms.
- Standardize pixel spacing: resample images to a consistent in-plane pixel spacing (e.g. 0.05–0.07 mm/px) using SimpleITK or sitk.ResampleImageFilter so models see consistent physical scales.
- Crop to breast: detect background (air) vs breast (threshold + connected components) and crop to breast region. Remove right/left padding and optionally remove pectoral muscle on MLO views using Hough/line or simple morphological/contour heuristics.
- Per-view normalization: compute per-view normalization statistics (mean/std or robust percentiles) and normalize each view separately. Avoid mixing CC/MLO distributions.
- Multi-scale: keep a high-resolution global image (e.g. long side 2000–3000 px) and extract patches for a local pathway centered on suspicious areas or dense-grid patches (512x512 or 1024x1024 depending on GPU memory).
- Maintain metadata: keep view, laterality, pixel spacing, acquisition info for later fusion and calibration.

Practical tips

- Use floating point for computations, but store preprocessed files in compressed 16-bit (uint16) or float32 NumPy/TFRecords if needed for I/O speed.
- Be conservative with downsampling — train global pathway on as-large-as-your-GPU memory allows and use crops for local features.
- Implement deterministic preprocessing pipelines (record random seeds and augmentation seeds for reproducibility).

2) Backbone & Initialization

- Backbones: ResNet-50 / ResNet-101, EfficientNet-B4, ConvNeXt-T/B are good choices. For mammography, larger receptive field and strong low-frequency modeling help.
- Initialization: use RadImageNet weights or other medical pretrained weights if available (improves convergence). If not available, use ImageNet pretraining + careful LR scheduling.
- Use libraries: timm for backbones, pretrained weights loader where available.

3) Architecture

- Global-local design (GMIC/GLAM style):
  - Global pathway: understand coarse context from full-view image.
  - Local pathway: take high-resolution patches (either detector proposals or uniform grid) and extract fine features.
  - Fuse: concatenate or attention-fuse global & local features per view.
- Multi-view fusion: combine CC/MLO of both breasts (4 views) with view-specific encoders or weight-sharing plus a fusion head. Consider positional embeddings for laterality/view.
- Case-level MIL: when only exam-level labels exist, use Multiple Instance Learning: aggregate patch/view scores (max, attention pooling) to produce exam prediction.

4) Training

- Losses: focal loss or class-balanced focal loss, try class-balanced (CB) loss for long-tailed distributions. Also experiment with label-smoothing and ensembling.
- Augmentations: strong augmentations but respect mammography physics — flips across the midline may be valid depending on laterality handling; avoid unrealistic intensity transforms that break diagnostic cues. Use MixUp or CutMix at image or patch level with care.
- Sampling: balance by case-level sampling (oversample positive cases) rather than individual images to avoid patient leakage.
- Curriculum: start training global-only, then add local pathway and MIL fine-tuning.
- Optimizer & schedule: AdamW with weight decay; cosine or step LR schedule. Use warmup for pretrained backbones.
- Mixed precision training (AMP) recommended for large images/patches.

5) Evaluation & Robustness

- External validation: test on different site/dataset (INbreast, CBIS-DDSM, CMMD, OMI-DB) to measure domain shift. Expect performance drop; use recalibration/temperature scaling.
- Metrics: ROC-AUC at exam and breast level, sensitivity at fixed specificity, FROC for lesion localization if annotations available.
- Calibration & Uncertainty: evaluate calibration (ECE) and use MC-dropout or test-time augmentation for uncertainty estimates.
- Explainability: use GradCAM / attention maps and patch scoring to localize suspicious areas for clinician review.

References & pointers

- GMIC/GLAM papers and public code (global-local fusion for mammography).
- RSNA and Nature reviews about mammography AI best practices.
- RadImageNet for pretrained weights.
- Datasets: INbreast, CBIS-DDSM, CMMD, OMI-DB (use for external testing and domain generalization checks).

"This recipe is a concise practical checklist — adapt hyperparameters, input sizes and losses to your hardware and dataset size."
