#!/usr/bin/env python3
"""
Collect and move dataset files into a single data/raw directory, writing a manifest.
Supports dry-run and optional backup tar creation.
"""
import argparse
import pathlib
import shutil
import hashlib
import csv
import datetime
import tarfile
import sys
import os

EXT = {'.png', '.jpg', '.jpeg', '.svs', '.dcm', '.tif', '.tiff', '.nii', '.nii.gz', '.csv', '.xlsx'}
here = pathlib.Path(__file__).resolve().parents[1]


def md5(path, chunk=8192):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def collect_files(roots):
    files = []
    for r in roots:
        p = (here / r).resolve()
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in EXT:
                files.append(p)
        else:
            for f in p.rglob('*'):
                if f.is_file() and f.suffix.lower() in EXT:
                    files.append(f)
    return files


def make_backup(roots, backup_path):
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup_path, "w:gz") as tar:
        for r in roots:
            p = (here / r).resolve()
            if p.exists():
                tar.add(str(p), arcname=str(p.relative_to(here)))
    return backup_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True, help='folders/files to move (relative to repo root)')
    ap.add_argument('--target', default='data/raw', help='target base directory (relative)')
    ap.add_argument('--manifest', default='data/manifest/move_manifest.csv')
    ap.add_argument('--backup', help='optional backup tar.gz path (relative)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--update-csv', help='path to CSV to update references (optional)')
    args = ap.parse_args()

    roots = args.roots
    target_base = (here / args.target).resolve()
    manifest_path = (here / args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backup:
        backup_path = (here / args.backup).resolve()
        print(f'Creating backup to {backup_path} ...')
        if not args.dry_run:
            make_backup(roots, backup_path)
        print('Backup done.')

    files = collect_files(roots)
    print(f'Found {len(files)} files to process.')

    rows = []
    for f in files:
        try:
            rel = f.relative_to(here)
        except Exception:
            rel = f
        new_path = target_base / rel
        rows.append((str(rel), str(new_path.relative_to(here)), f.stat().st_size, None))

    if args.dry_run:
        print('DRY RUN - actions that would be performed:')
        for orig, new, size, _ in rows[:1000]:
            print(f'{orig} -> {new} ({size} bytes)')
        print('Dry run complete.')
        sys.exit(0)

    # perform moves
    with open(str(manifest_path), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original_path', 'new_path', 'size_bytes', 'md5', 'moved_at'])
        for orig_rel, new_rel, size, _ in rows:
            orig = here / orig_rel
            new = here / new_rel
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(orig), str(new))
            m = md5(new)
            ts = datetime.datetime.utcnow().isoformat()
            writer.writerow([str(orig_rel), str(new_rel), size, m, ts])

    print(f'Moved {len(rows)} files. Manifest saved to {manifest_path}')

    # optional CSV path-update (simple substring replacement)
    if args.update_csv:
        csv_p = here / args.update_csv
        if csv_p.exists():
            txt = csv_p.read_text()
            for orig_rel, new_rel, _, _ in rows:
                txt = txt.replace(orig_rel, new_rel)
            csv_p.write_text(txt)
            print(f'Updated CSV references in {csv_p}')
        else:
            print(f'CSV to update not found: {csv_p}')


if __name__ == '__main__':
    main()
