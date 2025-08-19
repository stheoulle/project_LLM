#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import random

IMAGE_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff','.dcm','.svs'}


def find_image_files(root):
    root = Path(root)
    files = [p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='processed_metadata.csv')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--out_dir', default='0_classify_chat/data/tables')
    p.add_argument('--pct', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    meta_p = Path(args.metadata).resolve()
    raw_root = Path(args.raw).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dfm = pd.read_csv(meta_p)
    # common pathological label column
    label_col = None
    for c in dfm.columns:
        if 'Pathology' in c or 'pathology' in c:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError('Could not find pathology column in metadata')

    print('Found pathology column:', label_col)

    files = find_image_files(raw_root)
    print('Found', len(files), 'image files under', raw_root)

    # Build map from Image_name -> label. Be flexible about column names and matching.
    name_col = None
    # try common exact candidates first
    for c in dfm.columns:
        low = c.lower().replace(' ', '_')
        if low in ('image_name', 'imagename', 'image', 'file_name', 'filename', 'file'):
            name_col = c
            break
    # fallback: any column that contains 'image' or 'file'
    if name_col is None:
        for c in dfm.columns:
            if 'image' in c.lower() or 'file' in c.lower():
                name_col = c
                break
    if name_col is None:
        raise RuntimeError('Could not find an image name column (e.g. Image_name) in metadata')

    name_to_label = {}
    for _, r in dfm.iterrows():
        raw_name = r.get(name_col, None)
        if pd.isna(raw_name):
            continue
        name = str(raw_name).strip()
        if name == '':
            continue
        # also store stem (without extension) to help matching
        try:
            stem = Path(name).stem
        except Exception:
            stem = name
        lab = str(r[label_col]) if pd.notna(r[label_col]) else ''
        lab = lab.strip()
        binlab = 1 if ('Malignant' in lab or 'malignant' in lab or 'Malign' in lab) else 0
        name_to_label[name] = binlab
        if stem != name:
            name_to_label[stem] = binlab

    rows = []
    for f in files:
        fname = f.name
        fname_stem = f.stem
        matched = None
        # try exact filename, stem, or substring match
        for name in name_to_label.keys():
            if name == fname or name == fname_stem or name in fname:
                matched = name
                break
        if matched:
            # store path relative to data/raw (so dataloader can resolve root_dir/data/raw + rel)
            try:
                rel = f.relative_to(raw_root)
            except Exception:
                # fallback: relative to repo root
                rel = f
            # ensure POSIX style path string (consistent across platforms)
            rel_str = rel.as_posix() if isinstance(rel, Path) else str(rel)
            rows.append((rel_str, name_to_label[matched]))

    if len(rows)==0:
        raise RuntimeError('No matching image files found for metadata. Please ensure data/raw contains images and names match metadata Image_name.')

    total = len(rows)
    sample_n = max(1, int(len(rows) * args.pct))
    if sample_n > len(rows):
        sample_n = len(rows)
    random.seed(args.seed)
    sampled = random.sample(rows, sample_n)

    # split 80/20
    n_train = int(len(sampled) * 0.8)
    train = sampled[:n_train]
    val = sampled[n_train:]

    import csv
    train_csv = out_dir / '5pct_train.csv'
    val_csv = out_dir / '5pct_val.csv'
    with open(train_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path','label'])
        w.writerows(train)
    with open(val_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path','label'])
        w.writerows(val)

    print(f'Sampled {len(sampled)} files ({args.pct*100}%). Train={len(train)} Val={len(val)}')
    print('Wrote:', train_csv, val_csv)


if __name__ == '__main__':
    main()
