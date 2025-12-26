#!/usr/bin/env python3
"""
Dataset prep wizard for EC2 training pipeline.

Interactive walkthrough for preparing and uploading training jobs.

Usage:
    python prep.py
"""

import random
import shutil
import hashlib
import sys
import yaml
import boto3
from pathlib import Path


def main():
    """Launch the interactive wizard."""
    print("=" * 60)
    print("  EC2 YOLO Training - Job Setup Wizard")
    print("=" * 60)

    # Step 1: Mode selection
    print("\n[1/6] What do you want to do?\n")
    print("  1. Prep raw images + labels (need to split train/val)")
    print("  2. Merge multiple datasets (already have train/val splits)")
    print("  3. Upload existing YOLO dataset (already structured)")
    print()

    mode = input("Choose [1/2/3]: ").strip()
    while mode not in ['1', '2', '3']:
        mode = input("Please enter 1, 2, or 3: ").strip()

    if mode == '1':
        interactive_prep()
    elif mode == '2':
        interactive_merge()
    else:
        interactive_upload_existing()


def find_image_label_pairs(input_dir):
    """Find all image files that have corresponding label files."""
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    pairs = []
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                pairs.append((img_path, label_path))

    return pairs


def validate_labels(pairs, num_classes):
    """Check that all labels have valid class IDs."""
    errors = []
    for img_path, label_path in pairs:
        for line_num, line in enumerate(label_path.read_text().strip().split('\n'), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append(f"{label_path.name}:{line_num} - invalid format")
                continue
            class_id = int(parts[0])
            if class_id < 0 or class_id >= num_classes:
                errors.append(f"{label_path.name}:{line_num} - class {class_id} out of range")

    return errors


def prep_dataset(input_dir, classes, job_id, val_split=0.2, seed=42):
    """Prepare dataset from raw images + labels."""
    random.seed(seed)

    pairs = find_image_label_pairs(input_dir)

    # Validate
    errors = validate_labels(pairs, len(classes))
    if errors:
        print("\nLabel errors found:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        raise ValueError("Fix label errors before continuing")

    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create output structure
    job_dir = Path('./jobs') / job_id
    dataset_dir = job_dir / 'dataset'

    for split, split_pairs in [('train', train_pairs), ('valid', val_pairs)]:
        img_dir = dataset_dir / split / 'images'
        lbl_dir = dataset_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            shutil.copy(img_path, img_dir / img_path.name)
            shutil.copy(label_path, lbl_dir / label_path.name)

    # Create data.yaml
    data_yaml = {
        'path': '.',
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(classes),
        'names': classes,
    }
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n  Created: {job_dir}")
    print(f"  Train: {len(train_pairs)}, Valid: {len(val_pairs)}")

    return job_dir


def hash_image(img_path):
    """MD5 hash for deduplication."""
    return hashlib.md5(Path(img_path).read_bytes()).hexdigest()


def merge_datasets(dataset_dirs, job_id, seed=42):
    """Merge multiple YOLO-format datasets."""
    random.seed(seed)

    print(f"\nMerging {len(dataset_dirs)} datasets...")

    # First pass: collect all class names
    all_classes = {}
    dataset_configs = []

    for ds_path in dataset_dirs:
        ds_path = Path(ds_path)
        data_yaml = ds_path / 'data.yaml'

        if not data_yaml.exists():
            for f in ds_path.rglob('data.yaml'):
                data_yaml = f
                ds_path = f.parent
                break

        if not data_yaml.exists():
            print(f"  WARNING: No data.yaml in {ds_path}, skipping")
            continue

        with open(data_yaml) as f:
            config = yaml.safe_load(f)

        classes = config.get('names', [])
        if isinstance(classes, dict):
            classes = list(classes.values())

        print(f"  {ds_path.name}: {classes}")

        for cls in classes:
            all_classes[cls] = all_classes.get(cls, 0) + 1

        dataset_configs.append({
            'path': ds_path,
            'classes': classes,
        })

    if not dataset_configs:
        raise ValueError("No valid datasets found")

    unified_classes = sorted(all_classes.keys())
    print(f"\n  Unified: {unified_classes}")

    # Create output structure
    job_dir = Path('./jobs') / job_id
    dataset_dir = job_dir / 'dataset'

    for split in ['train', 'valid']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Second pass: copy and remap
    image_hashes = {}
    stats = {'images': 0, 'duplicates': 0}

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        ds_classes = ds_info['classes']

        class_remap = {}
        for old_id, cls_name in enumerate(ds_classes):
            class_remap[old_id] = unified_classes.index(cls_name)

        for split in ['train', 'valid']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue

                img_hash = hash_image(img_path)
                if img_hash in image_hashes:
                    stats['duplicates'] += 1
                    continue

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                # Remap labels
                new_lines = []
                for line in lbl_path.read_text().strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    old_class_id = int(parts[0])
                    new_class_id = class_remap.get(old_class_id)
                    if new_class_id is not None:
                        parts[0] = str(new_class_id)
                        new_lines.append(' '.join(parts))

                if not new_lines:
                    continue

                prefix = ds_path.name.replace('/', '_').replace(' ', '_')
                new_img_name = f"{prefix}_{img_path.name}"
                new_lbl_name = f"{prefix}_{img_path.stem}.txt"

                shutil.copy(img_path, dataset_dir / split / 'images' / new_img_name)
                (dataset_dir / split / 'labels' / new_lbl_name).write_text('\n'.join(new_lines))

                image_hashes[img_hash] = new_img_name
                stats['images'] += 1

    # Create data.yaml
    data_yaml = {
        'path': '.',
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(unified_classes),
        'names': unified_classes,
    }
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    train_count = len(list((dataset_dir / 'train' / 'images').iterdir()))
    valid_count = len(list((dataset_dir / 'valid' / 'images').iterdir()))

    print(f"\n  Images: {stats['images']} (removed {stats['duplicates']} duplicates)")
    print(f"  Train: {train_count}, Valid: {valid_count}")

    return job_dir


def upload_to_s3(job_dir, bucket):
    """Upload prepared job to S3."""
    s3 = boto3.client('s3')
    job_dir = Path(job_dir)
    job_id = job_dir.name

    print(f"\nUploading to s3://{bucket}/jobs/{job_id}/")

    file_count = 0
    for local_path in job_dir.rglob('*'):
        if local_path.is_file():
            relative_path = local_path.relative_to(job_dir)
            s3_key = f"jobs/{job_id}/{relative_path}"
            s3.upload_file(str(local_path), bucket, s3_key)
            file_count += 1

    print(f"  Uploaded {file_count} files")
    print(f"\nTraining will start automatically!")


def interactive_prep():
    """Interactive prep mode."""
    print("\n[2/6] Where are your images + labels?\n")
    print("  Expected: folder with image.jpg + image.txt pairs")

    input_dir = Path(input("\nPath: ").strip()).expanduser()

    while not input_dir.exists():
        print(f"  Not found: {input_dir}")
        input_dir = Path(input("Path: ").strip()).expanduser()

    pairs = find_image_label_pairs(input_dir)
    print(f"\n  Found {len(pairs)} image/label pairs")

    if not pairs:
        print("  No valid pairs found!")
        return

    # Classes
    print("\n[3/6] Class names (in order: class 0, class 1, ...)\n")
    print("  Example: chicken,egg")

    classes_str = input("\nClasses: ").strip()
    classes = [c.strip() for c in classes_str.split(',')]
    print(f"  {classes}")

    # Job ID
    print("\n[4/6] Job ID\n")
    print("  Example: chicken-detector-v1")

    job_id = input("\nJob ID: ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("Job ID: ").strip()

    # Val split
    print("\n[5/6] Validation split")
    val_split = input("\nRatio [0.2]: ").strip() or "0.2"
    val_split = float(val_split)

    # Config
    config = get_training_config()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Input:    {input_dir} ({len(pairs)} pairs)")
    print(f"  Classes:  {classes}")
    print(f"  Job ID:   {job_id}")
    print(f"  Split:    {1-val_split:.0%} train / {val_split:.0%} val")
    print(f"  Model:    {config['model']}")
    print(f"  Instance: {config['instance_type']}")
    print(f"  Epochs:   {config['epochs']}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    job_dir = prep_dataset(str(input_dir), classes, job_id, val_split)

    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    upload_prompt(job_dir)


def interactive_merge():
    """Interactive merge mode."""
    print("\n[2/6] Dataset directories to merge\n")
    print("  Enter paths one per line, empty line when done")

    datasets = []
    while True:
        path = input(f"\nDataset {len(datasets) + 1} (Enter to finish): ").strip()
        if not path:
            if len(datasets) < 2:
                print("  Need at least 2 datasets")
                continue
            break

        path = Path(path).expanduser()
        if not path.exists():
            print(f"  Not found: {path}")
            continue

        has_data_yaml = (path / 'data.yaml').exists() or list(path.rglob('data.yaml'))
        if not has_data_yaml:
            print(f"  No data.yaml in {path}")
            continue

        datasets.append(str(path))
        print(f"  Added: {path.name}")

    # Job ID
    print("\n[3/6] Job ID")
    job_id = input("\nJob ID: ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("Job ID: ").strip()

    # Config
    config = get_training_config()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Datasets: {len(datasets)}")
    for ds in datasets:
        print(f"            - {Path(ds).name}")
    print(f"  Job ID:   {job_id}")
    print(f"  Model:    {config['model']}")
    print(f"  Instance: {config['instance_type']}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    job_dir = merge_datasets(datasets, job_id)

    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    upload_prompt(job_dir)


def interactive_upload_existing():
    """Upload existing YOLO dataset."""
    print("\n[2/6] Dataset location\n")
    print("  Expected: folder with data.yaml, train/, valid/")

    dataset_dir = Path(input("\nPath: ").strip()).expanduser()

    while not (dataset_dir / 'data.yaml').exists():
        found = list(dataset_dir.rglob('data.yaml'))
        if found:
            dataset_dir = found[0].parent
            print(f"  Found data.yaml in: {dataset_dir}")
            break
        print(f"  No data.yaml in {dataset_dir}")
        dataset_dir = Path(input("Path: ").strip()).expanduser()

    # Job ID
    print("\n[3/6] Job ID")
    job_id = input("\nJob ID: ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("Job ID: ").strip()

    # Config
    config = get_training_config()

    # Create job
    job_dir = Path('./jobs') / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    dest = job_dir / 'dataset'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(dataset_dir, dest)

    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n  Created: {job_dir}")

    upload_prompt(job_dir)


def get_training_config():
    """Get training configuration interactively."""
    print("\n[5/6] Training config\n")

    print("  Models:")
    print("    1. yolo12n.pt  (nano)")
    print("    2. yolo12s.pt  (small)")
    print("    3. yolo12m.pt  (medium) *")
    print("    4. yolo12l.pt  (large)")
    print("    5. yolo12x.pt  (xlarge)")

    model_choice = input("\nModel [3]: ").strip() or "3"
    models = {'1': 'yolo12n.pt', '2': 'yolo12s.pt', '3': 'yolo12m.pt',
              '4': 'yolo12l.pt', '5': 'yolo12x.pt'}
    model = models.get(model_choice, 'yolo12m.pt')

    print("\n  Instances:")
    print("    1. g5.xlarge   ($1/hr, A10G 24GB) *")
    print("    2. g5.2xlarge  ($1.50/hr, A10G)")
    print("    3. g4dn.xlarge ($0.50/hr, T4 16GB)")

    inst_choice = input("\nInstance [1]: ").strip() or "1"
    instances = {'1': 'g5.xlarge', '2': 'g5.2xlarge', '3': 'g4dn.xlarge'}
    instance_type = instances.get(inst_choice, 'g5.xlarge')

    epochs = int(input("\nEpochs [120]: ").strip() or "120")
    batch = int(input("Batch size [16]: ").strip() or "16")

    return {
        'model': model,
        'instance_type': instance_type,
        'epochs': epochs,
        'batch': batch,
        'imgsz': 640,
        'patience': 20,
    }


def upload_prompt(job_dir):
    """Prompt for S3 upload."""
    print("\n[6/6] Upload to S3?\n")

    if input("Upload now? [y/N]: ").strip().lower() != 'y':
        print(f"\nJob ready at: {job_dir}")
        return

    bucket = input("\nS3 bucket: ").strip()
    while not bucket:
        bucket = input("S3 bucket: ").strip()

    upload_to_s3(job_dir, bucket)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
