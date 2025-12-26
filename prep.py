#!/usr/bin/env python3
"""
Dataset prep script for EC2 training pipeline.

Two modes:
1. PREP MODE: Takes raw images + labels, splits train/val, creates structure
2. MERGE MODE: Combines multiple YOLO datasets (e.g., from Roboflow exports)

Usage:
    # Prep raw images + labels
    python prep.py ./my_images --classes chicken,egg --job-id chicken-v1

    # Merge multiple Roboflow exports
    python prep.py merge ./dataset1 ./dataset2 ./dataset3 --job-id combined-v1

    # Upload after prep/merge
    python prep.py ... --upload --bucket my-bucket
"""

import argparse
import random
import shutil
import hashlib
import sys
import yaml
import boto3
from pathlib import Path
from collections import Counter


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
            else:
                print(f"  Skipping {img_path.name} (no label file)")

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
                errors.append(f"{label_path.name}:{line_num} - class {class_id} out of range (0-{num_classes-1})")

    return errors


def prep_dataset(input_dir, classes, job_id, val_split=0.2, output_dir='./jobs', seed=42):
    """Prepare dataset for training pipeline."""
    random.seed(seed)

    print(f"Preparing dataset: {job_id}")
    print(f"  Classes: {classes}")
    print(f"  Validation split: {val_split:.0%}")

    # Find pairs
    pairs = find_image_label_pairs(input_dir)
    if not pairs:
        raise ValueError(f"No image/label pairs found in {input_dir}")

    print(f"  Found {len(pairs)} image/label pairs")

    # Validate labels
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

    print(f"  Train: {len(train_pairs)}, Valid: {len(val_pairs)}")

    # Create output structure
    job_dir = Path(output_dir) / job_id
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

    # Create config.yaml with sensible defaults
    config = {
        'model': 'yolo12m.pt',
        'instance_type': 'g5.xlarge',
        'epochs': 120,
        'batch': 16,
        'imgsz': 640,
        'patience': 20,
    }
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nCreated job at: {job_dir}")
    print(f"  {job_dir}/config.yaml")
    print(f"  {job_dir}/dataset/data.yaml")
    print(f"  {job_dir}/dataset/train/  ({len(train_pairs)} images)")
    print(f"  {job_dir}/dataset/valid/  ({len(val_pairs)} images)")

    return job_dir


def hash_image(img_path):
    """MD5 hash for deduplication."""
    return hashlib.md5(Path(img_path).read_bytes()).hexdigest()


def merge_datasets(dataset_dirs, job_id, output_dir='./jobs', seed=42):
    """
    Merge multiple YOLO-format datasets (e.g., Roboflow exports).

    - Unifies class names across datasets
    - Deduplicates images by content hash
    - Remaps class IDs to unified scheme
    """
    random.seed(seed)

    print(f"Merging {len(dataset_dirs)} datasets into: {job_id}")

    # First pass: collect all class names
    all_classes = {}  # class_name -> count
    dataset_configs = []

    for ds_path in dataset_dirs:
        ds_path = Path(ds_path)
        data_yaml = ds_path / 'data.yaml'

        if not data_yaml.exists():
            # Try subdirectory (some exports have extra nesting)
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

        print(f"  {ds_path.name}: {len(classes)} classes - {classes}")

        for cls in classes:
            all_classes[cls] = all_classes.get(cls, 0) + 1

        dataset_configs.append({
            'path': ds_path,
            'classes': classes,
            'config': config,
        })

    if not dataset_configs:
        raise ValueError("No valid datasets found")

    # Create unified class list (sorted for consistency)
    unified_classes = sorted(all_classes.keys())
    print(f"\nUnified classes ({len(unified_classes)}): {unified_classes}")

    # Create output structure
    job_dir = Path(output_dir) / job_id
    dataset_dir = job_dir / 'dataset'

    for split in ['train', 'valid']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Second pass: copy and remap
    image_hashes = {}
    stats = {'images': 0, 'duplicates': 0, 'annotations': 0}

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        ds_classes = ds_info['classes']

        # Build class ID remap: old_id -> new_id
        class_remap = {}
        for old_id, cls_name in enumerate(ds_classes):
            class_remap[old_id] = unified_classes.index(cls_name)

        # Process train and valid splits
        for split in ['train', 'valid']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue

                # Check for duplicate
                img_hash = hash_image(img_path)
                if img_hash in image_hashes:
                    stats['duplicates'] += 1
                    continue

                # Find label file
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
                        stats['annotations'] += 1

                if not new_lines:
                    continue

                # Copy with prefixed name to avoid collisions
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

    # Create config.yaml
    config = {
        'model': 'yolo12m.pt',
        'instance_type': 'g5.xlarge',
        'epochs': 120,
        'batch': 16,
        'imgsz': 640,
        'patience': 20,
    }
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Count per split
    train_count = len(list((dataset_dir / 'train' / 'images').iterdir()))
    valid_count = len(list((dataset_dir / 'valid' / 'images').iterdir()))

    print(f"\nMerge complete:")
    print(f"  Total images: {stats['images']} (removed {stats['duplicates']} duplicates)")
    print(f"  Total annotations: {stats['annotations']}")
    print(f"  Train: {train_count}, Valid: {valid_count}")
    print(f"  Output: {job_dir}")

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
    print(f"\nTo start training, the Lambda will trigger automatically.")
    print(f"Or manually re-upload config.yaml to trigger:")
    print(f"  aws s3 cp {job_dir}/config.yaml s3://{bucket}/jobs/{job_id}/config.yaml")


def interactive_mode():
    """Interactive TUI walkthrough for creating a training job."""
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
        return interactive_prep()
    elif mode == '2':
        return interactive_merge()
    else:
        return interactive_upload_existing()


def interactive_prep():
    """Interactive prep mode."""
    print("\n[2/6] Where are your images + labels?\n")
    print("  Expected: folder with image.jpg + image.txt pairs")
    print()

    input_dir = input("Path to folder: ").strip()
    input_dir = Path(input_dir).expanduser()

    while not input_dir.exists():
        print(f"  Directory not found: {input_dir}")
        input_dir = Path(input("Path to folder: ").strip()).expanduser()

    # Count pairs
    pairs = find_image_label_pairs(input_dir)
    print(f"\n  Found {len(pairs)} image/label pairs")

    if not pairs:
        print("  No valid pairs found. Check that .txt labels exist.")
        return None

    # Classes
    print("\n[3/6] What classes are in your labels?\n")
    print("  Enter comma-separated class names in order (class 0, class 1, ...)")
    print("  Example: chicken,egg")
    print()

    classes_str = input("Classes: ").strip()
    classes = [c.strip() for c in classes_str.split(',')]
    print(f"\n  Classes: {classes}")

    # Job ID
    print("\n[4/6] Give this job a unique ID\n")
    print("  Example: chicken-detector-v1")
    print()

    job_id = input("Job ID: ").strip()
    while not job_id or ' ' in job_id:
        print("  Job ID cannot be empty or contain spaces")
        job_id = input("Job ID: ").strip()

    # Val split
    print("\n[5/6] Train/validation split\n")
    val_split = input("Validation ratio [0.2]: ").strip() or "0.2"
    val_split = float(val_split)

    # Training config
    config = interactive_training_config()

    # Confirm
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Input:      {input_dir}")
    print(f"  Pairs:      {len(pairs)}")
    print(f"  Classes:    {classes}")
    print(f"  Job ID:     {job_id}")
    print(f"  Val split:  {val_split:.0%}")
    print(f"  Model:      {config['model']}")
    print(f"  Instance:   {config['instance_type']}")
    print(f"  Epochs:     {config['epochs']}")
    print()

    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Cancelled.")
        return None

    # Run prep
    job_dir = prep_dataset(
        input_dir=str(input_dir),
        classes=classes,
        job_id=job_id,
        val_split=val_split,
    )

    # Update config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return interactive_upload_prompt(job_dir)


def interactive_merge():
    """Interactive merge mode."""
    print("\n[2/6] Enter dataset directories to merge\n")
    print("  Enter paths one per line, empty line when done")
    print("  Example: ./dataset1")
    print()

    datasets = []
    while True:
        path = input(f"Dataset {len(datasets) + 1} (or Enter to finish): ").strip()
        if not path:
            if len(datasets) < 2:
                print("  Need at least 2 datasets to merge")
                continue
            break

        path = Path(path).expanduser()
        if not path.exists():
            print(f"  Not found: {path}")
            continue

        # Check for data.yaml
        if not (path / 'data.yaml').exists():
            found = list(path.rglob('data.yaml'))
            if not found:
                print(f"  No data.yaml found in {path}")
                continue

        datasets.append(str(path))
        print(f"  Added: {path}")

    print(f"\n  Merging {len(datasets)} datasets")

    # Job ID
    print("\n[3/6] Give this job a unique ID\n")
    job_id = input("Job ID: ").strip()
    while not job_id or ' ' in job_id:
        print("  Job ID cannot be empty or contain spaces")
        job_id = input("Job ID: ").strip()

    # Training config
    config = interactive_training_config()

    # Confirm
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Datasets:   {len(datasets)}")
    for ds in datasets:
        print(f"              - {ds}")
    print(f"  Job ID:     {job_id}")
    print(f"  Model:      {config['model']}")
    print(f"  Instance:   {config['instance_type']}")
    print(f"  Epochs:     {config['epochs']}")
    print()

    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Cancelled.")
        return None

    # Run merge
    job_dir = merge_datasets(
        dataset_dirs=datasets,
        job_id=job_id,
    )

    # Update config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return interactive_upload_prompt(job_dir)


def interactive_upload_existing():
    """Upload an already-structured YOLO dataset."""
    print("\n[2/6] Where is your dataset?\n")
    print("  Expected structure:")
    print("    dataset/")
    print("      data.yaml")
    print("      train/images/, train/labels/")
    print("      valid/images/, valid/labels/")
    print()

    dataset_dir = Path(input("Path to dataset: ").strip()).expanduser()

    while not (dataset_dir / 'data.yaml').exists():
        print(f"  No data.yaml found in {dataset_dir}")
        dataset_dir = Path(input("Path to dataset: ").strip()).expanduser()

    # Job ID
    print("\n[3/6] Give this job a unique ID\n")
    job_id = input("Job ID: ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("Job ID: ").strip()

    # Training config
    config = interactive_training_config()

    # Create job structure
    job_dir = Path('./jobs') / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset
    dest_dataset = job_dir / 'dataset'
    if dest_dataset.exists():
        shutil.rmtree(dest_dataset)
    shutil.copytree(dataset_dir, dest_dataset)

    # Write config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n  Created job at: {job_dir}")

    return interactive_upload_prompt(job_dir)


def interactive_training_config():
    """Interactive training configuration."""
    print("\n[5/6] Training configuration\n")

    # Model
    print("  Available models:")
    print("    1. yolo12n.pt  (nano - fastest, least accurate)")
    print("    2. yolo12s.pt  (small)")
    print("    3. yolo12m.pt  (medium - recommended)")
    print("    4. yolo12l.pt  (large)")
    print("    5. yolo12x.pt  (xlarge - slowest, most accurate)")
    print()

    model_choice = input("Model [3]: ").strip() or "3"
    models = {
        '1': 'yolo12n.pt', '2': 'yolo12s.pt', '3': 'yolo12m.pt',
        '4': 'yolo12l.pt', '5': 'yolo12x.pt'
    }
    model = models.get(model_choice, 'yolo12m.pt')

    # Instance type
    print("\n  Instance types (GPU):")
    print("    1. g5.xlarge   (~$1.00/hr - 1x A10G, 24GB)")
    print("    2. g5.2xlarge  (~$1.50/hr - 1x A10G, 24GB, more CPU)")
    print("    3. g4dn.xlarge (~$0.50/hr - 1x T4, 16GB)")
    print()

    instance_choice = input("Instance [1]: ").strip() or "1"
    instances = {
        '1': 'g5.xlarge', '2': 'g5.2xlarge', '3': 'g4dn.xlarge'
    }
    instance_type = instances.get(instance_choice, 'g5.xlarge')

    # Epochs
    epochs = input("\nEpochs [120]: ").strip() or "120"
    epochs = int(epochs)

    # Batch size
    batch = input("Batch size [16]: ").strip() or "16"
    batch = int(batch)

    config = {
        'model': model,
        'instance_type': instance_type,
        'epochs': epochs,
        'batch': batch,
        'imgsz': 640,
        'patience': 20,
    }

    return config


def interactive_upload_prompt(job_dir):
    """Prompt to upload to S3."""
    print("\n[6/6] Upload to S3?\n")

    upload = input("Upload now? [y/N]: ").strip().lower()
    if upload != 'y':
        print(f"\nJob ready at: {job_dir}")
        print(f"\nTo upload later:")
        print(f"  python prep.py upload {job_dir} --bucket YOUR_BUCKET")
        return job_dir

    bucket = input("S3 bucket name: ").strip()
    while not bucket:
        bucket = input("S3 bucket name: ").strip()

    upload_to_s3(job_dir, bucket)
    return job_dir


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for EC2 training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive wizard (recommended)
    python prep.py

    # Prep raw images + labels
    python prep.py ./chicken_photos --classes chicken --job-id chicken-v1

    # Merge multiple Roboflow exports
    python prep.py merge ./dataset1 ./dataset2 --job-id combined-v1

    # Upload existing job
    python prep.py upload ./jobs/my-job --bucket my-bucket
        """
    )

    subparsers = parser.add_subparsers(dest='command')

    # Merge subcommand
    merge_parser = subparsers.add_parser('merge', help='Merge multiple YOLO datasets')
    merge_parser.add_argument('datasets', nargs='+', help='Dataset directories to merge')
    merge_parser.add_argument('--job-id', required=True, help='Unique job identifier')
    merge_parser.add_argument('--output-dir', default='./jobs', help='Output directory')
    merge_parser.add_argument('--upload', action='store_true', help='Upload to S3 after merge')
    merge_parser.add_argument('--bucket', help='S3 bucket name')
    merge_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Upload subcommand
    upload_parser = subparsers.add_parser('upload', help='Upload a prepared job to S3')
    upload_parser.add_argument('job_dir', help='Job directory to upload')
    upload_parser.add_argument('--bucket', required=True, help='S3 bucket name')

    # Default: prep mode (for backwards compatibility, input_dir is positional)
    parser.add_argument('input_dir', nargs='?', help='Directory with images + YOLO labels')
    parser.add_argument('--classes', help='Comma-separated class names')
    parser.add_argument('--job-id', help='Unique job identifier')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split (default: 0.2)')
    parser.add_argument('--output-dir', default='./jobs', help='Output directory')
    parser.add_argument('--upload', action='store_true', help='Upload to S3')
    parser.add_argument('--bucket', help='S3 bucket name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # No args = interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return

    if args.command == 'upload':
        upload_to_s3(args.job_dir, args.bucket)
        return

    if args.command == 'merge':
        if args.upload and not args.bucket:
            parser.error("--bucket is required when using --upload")
        job_dir = merge_datasets(
            dataset_dirs=args.datasets,
            job_id=args.job_id,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        if args.upload:
            upload_to_s3(job_dir, args.bucket)
        return

    # Prep mode
    if args.upload and not args.bucket:
        parser.error("--bucket is required when using --upload")

    if not args.input_dir:
        # No input_dir and not a subcommand = interactive
        interactive_mode()
        return

    if not args.classes:
        parser.error("--classes is required")
    if not args.job_id:
        parser.error("--job-id is required")

    classes = [c.strip() for c in args.classes.split(',')]
    job_dir = prep_dataset(
        input_dir=args.input_dir,
        classes=classes,
        job_id=args.job_id,
        val_split=args.val_split,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.upload:
        upload_to_s3(job_dir, args.bucket)


if __name__ == '__main__':
    main()
