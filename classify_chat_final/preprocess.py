#!/usr/bin/env python3
"""
Preprocessing utilities for mammography DICOMs (step 1 of the recipe).
- preserves bit depth (applies RescaleSlope/Intercept then works in float32)
- standardizes in-plane pixel spacing (resample to target spacing)
- crops to breast using Otsu + largest connected component
- optional pectoral muscle removal for MLO views (Hough-based heuristic)
- per-image normalization (robust percentiles)
- optional sliding-window patch extraction

Usage example:
    python preprocess.py --input_dir /path/to/study --output_dir /path/to/out --spacing 0.05 --patch_size 512 --patch_stride 256

Requires: pydicom, numpy, scikit-image, scipy
"""

from pathlib import Path
import argparse
import json
import os

import numpy as np
try:
    import pydicom
except Exception:
    raise RuntimeError("pydicom is required. Install with: pip install pydicom")

try:
    from skimage.transform import resize
    from skimage.filters import threshold_otsu
    from skimage import morphology
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
except Exception:
    raise RuntimeError("scikit-image is required. Install with: pip install scikit-image")

from scipy import ndimage as ndi


def read_dicom_image(path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept
    # get metadata
    px = None
    try:
        ps = getattr(ds, 'PixelSpacing', None)
        if ps is not None:
            px = float(ps[0])
    except Exception:
        px = None
    bits = getattr(ds, 'BitsAllocated', None)
    meta = {
        'path': str(path),
        'Rows': int(getattr(ds, 'Rows', 0)),
        'Columns': int(getattr(ds, 'Columns', 0)),
        'PixelSpacing': px,
        'BitsAllocated': int(bits) if bits is not None else None,
        'ViewPosition': getattr(ds, 'ViewPosition', None),
        'ViewCodeSequence': getattr(ds, 'ViewCodeSequence', None),
    }
    return arr, meta


def resample_to_spacing(img, orig_spacing, target_spacing):
    """Resample 2D image to target in-plane spacing. sp in mm/px or same units.
    If orig_spacing is None, returns image unchanged.
    """
    if orig_spacing is None or target_spacing is None:
        return img
    scale = orig_spacing / float(target_spacing)
    if scale == 1.0:
        return img
    new_shape = (int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale)))
    if new_shape[0] < 1 or new_shape[1] < 1:
        raise ValueError("Invalid target spacing resulting in empty image")
    img_res = resize(img, new_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return img_res


def crop_to_breast(img, pad=16):
    # Compute a threshold and take largest connected component
    try:
        thr = threshold_otsu(img)
    except Exception:
        thr = np.percentile(img, 50)
    mask = img > thr * 0.5  # conservative
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n == 0:
        # fallback: return original
        h, w = img.shape
        return img, (0, 0, h, w), mask
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    largest = (sizes.argmax() + 1)
    breast_mask = labels == largest
    coords = np.argwhere(breast_mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    # add padding and clamp
    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, img.shape[0])
    x1 = min(x1 + pad, img.shape[1])
    cropped = img[y0:y1, x0:x1]
    return cropped, (y0, x0, y1, x1), breast_mask


def remove_pectoral_muscle(img):
    """Attempt to remove pectoral muscle by detecting a strong linear border near the upper corner.
    This is a heuristic and may fail; in that case we return the original mask.
    """
    try:
        edges = canny(img, sigma=3)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=50, line_gap=10)
        h, w = img.shape
        if not lines:
            return img
        # pick line closest to top-left or top-right corner by y-intercept
        best = None
        best_dist = float('inf')
        for (p0, p1) in lines:
            # compute distance from corner
            dist = min(np.hypot(p0[0], p0[1]), np.hypot(p1[0], p1[1]))
            if dist < best_dist:
                best_dist = dist
                best = (p0, p1)
        if best is None:
            return img
        (y0, x0), (y1, x1) = best
        # compute line equation ax + by + c = 0
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        Y, X = np.mgrid[0:h, 0:w]
        # mask one side of the line (the side near the corner)
        side = a * X + b * Y + c
        # choose side that contains the corner (0,0)
        corner_val = a * 0 + b * 0 + c
        if corner_val < 0:
            mask = side < 0
        else:
            mask = side > 0
        # zero out masked region
        out = img.copy()
        out[mask] = np.min(img)
        return out
    except Exception:
        return img


def normalize_image(img, method='percentile'):
    if method == 'percentile':
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
        if hi - lo <= 0:
            return np.clip((img - lo), 0, None).astype(np.float32)
        out = (img - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    elif method == 'zscore':
        m = np.mean(img)
        s = np.std(img)
        if s <= 0:
            return (img - m).astype(np.float32)
        return ((img - m) / s).astype(np.float32)
    else:
        return img.astype(np.float32)


def extract_patches(img, patch_size=512, stride=256):
    h, w = img.shape
    patches = []
    positions = []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            p = img[y:y + patch_size, x:x + patch_size]
            if p.shape != (patch_size, patch_size):
                # pad
                out = np.zeros((patch_size, patch_size), dtype=img.dtype)
                out[:p.shape[0], :p.shape[1]] = p
                p = out
            patches.append(p)
            positions.append((y, x))
    # also include bottom/right patches to cover edges
    if h % stride != 0 and h > patch_size:
        y = h - patch_size
        for x in range(0, max(1, w - patch_size + 1), stride):
            p = img[y:y + patch_size, x:x + patch_size]
            if p.shape != (patch_size, patch_size):
                out = np.zeros((patch_size, patch_size), dtype=img.dtype)
                out[:p.shape[0], :p.shape[1]] = p
                p = out
            patches.append(p)
            positions.append((y, x))
    if w % stride != 0 and w > patch_size:
        x = w - patch_size
        for y in range(0, max(1, h - patch_size + 1), stride):
            p = img[y:y + patch_size, x:x + patch_size]
            if p.shape != (patch_size, patch_size):
                out = np.zeros((patch_size, patch_size), dtype=img.dtype)
                out[:p.shape[0], :p.shape[1]] = p
                p = out
            patches.append(p)
            positions.append((y, x))
    # corner
    if h > patch_size and w > patch_size and (h % stride != 0 or w % stride != 0):
        y = h - patch_size
        x = w - patch_size
        p = img[y:y + patch_size, x:x + patch_size]
        patches.append(p)
        positions.append((y, x))
    return patches, positions


def process_directory(input_dir, output_dir, target_spacing=0.05, patch_size=None, patch_stride=None, remove_pectoral=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in input_dir.glob('**/*') if p.is_file()]
    dicoms = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            dicoms.append(p)
        except Exception:
            continue
    if not dicoms:
        raise RuntimeError('No DICOM files found in input_dir')

    manifest = []
    for p in dicoms:
        print('Processing', p)
        img, meta = read_dicom_image(p)
        orig_spacing = meta.get('PixelSpacing', None)
        img_rs = resample_to_spacing(img, orig_spacing, target_spacing)
        # crop
        cropped, bbox, mask = crop_to_breast(img_rs)
        if remove_pectoral:
            cropped = remove_pectoral_muscle(cropped)
        norm = normalize_image(cropped, method='percentile')
        # save full preprocessed image as npy and metadata
        stem = p.stem
        out_base = output_dir / stem
        np.save(str(out_base.with_suffix('.npy')), norm.astype(np.float32))
        meta_out = {
            'orig_path': str(p),
            'bbox': bbox,
            'orig_spacing': orig_spacing,
            'target_spacing': target_spacing,
            'shape': norm.shape,
            'metadata': meta,
        }
        with open(str(out_base.with_suffix('.json')), 'w') as f:
            json.dump(meta_out, f)
        rec = {'file': str(out_base.with_suffix('.npy')), 'meta': meta_out}
        # patches
        if patch_size is not None and patch_stride is not None:
            patches_dir = output_dir / f'{stem}_patches'
            patches_dir.mkdir(exist_ok=True)
            patches, positions = extract_patches(norm, patch_size=patch_size, stride=patch_stride)
            for i, (patch, pos) in enumerate(zip(patches, positions)):
                np.save(str(patches_dir / f'patch_{i:04d}.npy'), patch.astype(np.float32))
            rec['patches'] = {'count': len(patches), 'dir': str(patches_dir)}
        manifest.append(rec)
    with open(str(output_dir / 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Done. Outputs in', output_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--spacing', type=float, default=0.05, help='target in-plane spacing (mm/px)')
    p.add_argument('--patch_size', type=int, default=None)
    p.add_argument('--patch_stride', type=int, default=None)
    p.add_argument('--remove_pectoral', action='store_true')
    args = p.parse_args()
    process_directory(args.input_dir, args.output_dir, target_spacing=args.spacing, patch_size=args.patch_size, patch_stride=args.patch_stride, remove_pectoral=args.remove_pectoral)


if __name__ == '__main__':
    main()
