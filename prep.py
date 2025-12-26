#!/usr/bin/env python3
"""
Dataset prep script for EC2 training pipeline.

Takes a folder of images + YOLO labels and prepares it for S3 upload.

Usage:
    python prep.py ./my_images --classes chicken,egg --job-id chicken-v1
    python prep.py ./my_images --classes chicken --job-id chicken-v1 --upload

Expected input structure (YOLO format):
    my_images/
        image1.jpg
        image1.txt      # YOLO format: class_id x_center y_center width height
        image2.jpg
        image2.txt
        ...

Output structure:
    jobs/chicken-v1/
        config.yaml
        dataset/
            data.yaml
            train/
                images/
                labels/
            valid/
                images/
                labels/
"""

import argparse
import random
import shutil
import yaml
import boto3
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for EC2 training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prep (creates ./jobs/chicken-v1/)
    python prep.py ./chicken_photos --classes chicken --job-id chicken-v1

    # Multiple classes
    python prep.py ./data --classes cat,dog,bird --job-id pets-v1

    # Prep and upload to S3
    python prep.py ./data --classes chicken --job-id chicken-v1 --upload --bucket my-training-bucket

    # Custom validation split
    python prep.py ./data --classes chicken --job-id chicken-v1 --val-split 0.15
        """
    )

    parser.add_argument('input_dir', help='Directory containing images and YOLO label files')
    parser.add_argument('--classes', required=True, help='Comma-separated class names (order matters!)')
    parser.add_argument('--job-id', required=True, help='Unique job identifier')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--output-dir', default='./jobs', help='Output directory (default: ./jobs)')
    parser.add_argument('--upload', action='store_true', help='Upload to S3 after prep')
    parser.add_argument('--bucket', help='S3 bucket name (required if --upload)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')

    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(',')]

    if args.upload and not args.bucket:
        parser.error("--bucket is required when using --upload")

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
