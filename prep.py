#!/usr/bin/env python3
"""
Dataset prep wizard for EC2 training pipeline.

Usage:
    python prep.py
"""

import shutil
import hashlib
import sys
import yaml
import boto3
from pathlib import Path


def main():
    print("=" * 60)
    print("  EC2 YOLO Training - Job Setup")
    print("=" * 60)

    # Collect datasets
    print("\nDatasets to upload (Enter when done):\n")

    datasets = []
    while True:
        prompt = f"Dataset {len(datasets) + 1}: " if datasets else "Dataset: "
        path = input(prompt).strip()

        if not path:
            if not datasets:
                print("  Need at least one dataset")
                continue
            break

        path = Path(path).expanduser()
        if not path.exists():
            print(f"  Not found: {path}")
            continue

        # Find data.yaml
        data_yaml = path / 'data.yaml'
        if not data_yaml.exists():
            found = list(path.rglob('data.yaml'))
            if found:
                path = found[0].parent
                print(f"  Found: {path}")
            else:
                print(f"  No data.yaml in {path}")
                continue

        datasets.append(path)

        # Show what we found
        with open(path / 'data.yaml') as f:
            config = yaml.safe_load(f)
        classes = config.get('names', [])
        if isinstance(classes, dict):
            classes = list(classes.values())
        print(f"  Added: {path.name} ({len(classes)} classes: {classes})")

    # Job ID
    print(f"\nJob ID (e.g., spaghetti-v1):")
    job_id = input("\n> ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("> ").strip()

    # Training config
    config = get_training_config()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    if len(datasets) == 1:
        print(f"  Dataset:  {datasets[0].name}")
    else:
        print(f"  Datasets: {len(datasets)} (will merge)")
        for ds in datasets:
            print(f"            - {ds.name}")
    print(f"  Job ID:   {job_id}")
    print(f"  Model:    {config['model']}")
    print(f"  Instance: {config['instance_type']}")
    print(f"  Epochs:   {config['epochs']}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    # Process
    if len(datasets) == 1:
        job_dir = copy_single_dataset(datasets[0], job_id)
    else:
        job_dir = merge_datasets(datasets, job_id)

    # Write config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Upload
    print("\nS3 bucket:")
    bucket = input("\n> ").strip()
    while not bucket:
        bucket = input("> ").strip()

    upload_to_s3(job_dir, bucket)


def copy_single_dataset(dataset_path, job_id):
    """Copy a single dataset to job structure."""
    job_dir = Path('./jobs') / job_id
    dest = job_dir / 'dataset'

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(dataset_path, dest)
    print(f"\n  Created: {job_dir}")

    return job_dir


def merge_datasets(dataset_paths, job_id):
    """Merge multiple YOLO datasets."""
    print(f"\nMerging {len(dataset_paths)} datasets...")

    # Collect classes from all datasets
    all_classes = {}
    dataset_configs = []

    for ds_path in dataset_paths:
        with open(ds_path / 'data.yaml') as f:
            config = yaml.safe_load(f)

        classes = config.get('names', [])
        if isinstance(classes, dict):
            classes = list(classes.values())

        for cls in classes:
            all_classes[cls] = all_classes.get(cls, 0) + 1

        dataset_configs.append({'path': ds_path, 'classes': classes})

    unified_classes = sorted(all_classes.keys())
    print(f"  Unified classes: {unified_classes}")

    # Create output
    job_dir = Path('./jobs') / job_id
    dataset_dir = job_dir / 'dataset'

    for split in ['train', 'valid']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy and remap
    image_hashes = {}
    stats = {'images': 0, 'duplicates': 0}

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        ds_classes = ds_info['classes']

        # Class ID remapping
        class_remap = {i: unified_classes.index(cls) for i, cls in enumerate(ds_classes)}

        for split in ['train', 'valid']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue

                # Deduplicate
                img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                if img_hash in image_hashes:
                    stats['duplicates'] += 1
                    continue

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                # Remap class IDs
                new_lines = []
                for line in lbl_path.read_text().strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    old_id = int(parts[0])
                    parts[0] = str(class_remap[old_id])
                    new_lines.append(' '.join(parts))

                if not new_lines:
                    continue

                # Copy with prefix
                prefix = ds_path.name.replace(' ', '_')
                shutil.copy(img_path, dataset_dir / split / 'images' / f"{prefix}_{img_path.name}")
                (dataset_dir / split / 'labels' / f"{prefix}_{img_path.stem}.txt").write_text('\n'.join(new_lines))

                image_hashes[img_hash] = True
                stats['images'] += 1

    # Create data.yaml
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump({
            'path': '.',
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(unified_classes),
            'names': unified_classes,
        }, f, default_flow_style=False)

    print(f"  Images: {stats['images']} (removed {stats['duplicates']} duplicates)")
    print(f"  Created: {job_dir}")

    return job_dir


def get_training_config():
    """Get training configuration."""
    print("\nTraining config:\n")

    # Model: generation + size
    print("  Model (e.g., 12m, 11s, 8x):")
    model_input = input("  [12m]: ").strip().lower() or "12m"

    # Parse generation and size
    gen = ''.join(c for c in model_input if c.isdigit()) or '12'
    size = ''.join(c for c in model_input if c.isalpha()) or 'm'
    if size not in ['n', 's', 'm', 'l', 'x']:
        size = 'm'
    model = f"yolo{gen}{size}.pt"

    # Instance
    print("\n  Instance: 1=g5.xlarge 2=g5.2xlarge 3=g4dn.xlarge")
    inst_choice = input("  [1]: ").strip() or "1"
    instances = {'1': 'g5.xlarge', '2': 'g5.2xlarge', '3': 'g4dn.xlarge'}
    instance_type = instances.get(inst_choice, 'g5.xlarge')

    epochs = int(input("\n  Epochs [120]: ").strip() or "120")
    batch = int(input("  Batch [16]: ").strip() or "16")

    return {
        'model': model,
        'instance_type': instance_type,
        'epochs': epochs,
        'batch': batch,
        'imgsz': 640,
        'patience': 20,
    }


def upload_to_s3(job_dir, bucket):
    """Upload job to S3."""
    s3 = boto3.client('s3')
    job_id = job_dir.name

    print(f"\nUploading to s3://{bucket}/jobs/{job_id}/")

    count = 0
    for path in job_dir.rglob('*'):
        if path.is_file():
            key = f"jobs/{job_id}/{path.relative_to(job_dir)}"
            s3.upload_file(str(path), bucket, key)
            count += 1

    print(f"  Uploaded {count} files")
    print(f"\nTraining will start automatically!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
