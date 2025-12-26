#!/usr/bin/env python3
"""
EC2 YOLO Training - Dataset prep and job launcher.

Usage:
    python prep.py
"""

import base64
import shutil
import hashlib
import sys
import yaml
import boto3
from pathlib import Path

CONFIG_FILE = Path.home() / '.ec2-trainer.yaml'


def main():
    print("=" * 60)
    print("  EC2 YOLO Training")
    print("=" * 60)

    # Load or create infrastructure config
    infra = load_infra_config()

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
    print("\nJob ID (e.g., spaghetti-v1):")
    job_id = input("\n> ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("> ").strip()

    # Training config
    config = get_training_config()

    # S3 bucket
    bucket = pick_bucket(infra)

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
    print(f"  Bucket:   {bucket}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    # Check if job already exists
    if job_exists(bucket, job_id):
        print(f"\n  Warning: Job '{job_id}' already exists in S3!")
        if input("  Overwrite? [y/N]: ").strip().lower() != 'y':
            print("Cancelled.")
            return

    # Process datasets
    if len(datasets) == 1:
        job_dir = copy_single_dataset(datasets[0], job_id)
    else:
        job_dir = merge_datasets(datasets, job_id)

    # Write config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Upload to S3
    upload_to_s3(job_dir, bucket, job_id)

    # Create spot request
    create_spot_request(job_id, config['instance_type'], bucket, infra)

    print("\nTraining started! You'll get an SMS when it's done.")


def load_infra_config():
    """Load or create infrastructure config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        print(f"\nUsing config from {CONFIG_FILE}")
        return config

    print("\nFirst run - need AWS infrastructure config.")
    print("(This will be saved to ~/.ec2-trainer.yaml)\n")

    config = {
        'efs_id': input("EFS ID (fs-xxxxx): ").strip(),
        'subnet_id': input("Subnet ID (subnet-xxxxx): ").strip(),
        'security_group_id': input("Security Group ID (sg-xxxxx): ").strip(),
        'iam_instance_profile': input("IAM Instance Profile name: ").strip(),
        'ami_id': input("AMI ID [ami-0c7217cdde317cfec]: ").strip() or 'ami-0c7217cdde317cfec',
    }

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)
    print(f"\nSaved to {CONFIG_FILE}")

    return config


def pick_bucket(infra):
    """Pick S3 bucket (remembers last used)."""
    saved = infra.get('bucket')
    if saved:
        print(f"\nS3 bucket [{saved}]:")
        choice = input("> ").strip()
        if not choice:
            return saved
        if choice.isdigit():
            # User wants to pick from list
            pass
        else:
            return save_bucket(choice)

    print("\nS3 bucket:")
    try:
        response = boto3.client('s3').list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        if buckets:
            for i, b in enumerate(buckets, 1):
                print(f"  {i}. {b}")
            choice = input("\n> ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(buckets):
                return save_bucket(buckets[int(choice) - 1])
            return save_bucket(choice or buckets[0])
    except Exception:
        pass
    bucket = input("\n> ").strip()
    while not bucket:
        bucket = input("> ").strip()
    return save_bucket(bucket)


def save_bucket(bucket):
    """Save bucket to config for next time."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f) or {}
        config['bucket'] = bucket
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
    return bucket


def job_exists(bucket, job_id):
    """Check if job already exists in S3."""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"jobs/{job_id}/",
            MaxKeys=1
        )
        return response.get('KeyCount', 0) > 0
    except Exception:
        return False


def get_training_config():
    """Get training configuration."""
    print("\nTraining config:\n")

    # Model: generation + size
    print("  Model (e.g., 12m, 11s, 8x):")
    model_input = input("  [12m]: ").strip().lower() or "12m"

    gen = ''.join(c for c in model_input if c.isdigit()) or '12'
    size = ''.join(c for c in model_input if c.isalpha()) or 'm'
    if size not in ['n', 's', 'm', 'l', 'x']:
        size = 'm'
    model = f"yolo{gen}{size}.pt"

    print("\n  Instance (common: g5.xlarge, g5.2xlarge, g4dn.xlarge, p3.2xlarge):")
    instance_type = input("  [g5.xlarge]: ").strip() or "g5.xlarge"

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

    job_dir = Path('./jobs') / job_id
    dataset_dir = job_dir / 'dataset'

    for split in ['train', 'valid']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    image_hashes = {}
    stats = {'images': 0, 'duplicates': 0}

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        ds_classes = ds_info['classes']

        class_remap = {i: unified_classes.index(cls) for i, cls in enumerate(ds_classes)}

        for split in ['train', 'valid']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue

                img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                if img_hash in image_hashes:
                    stats['duplicates'] += 1
                    continue

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

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

                prefix = ds_path.name.replace(' ', '_')
                shutil.copy(img_path, dataset_dir / split / 'images' / f"{prefix}_{img_path.name}")
                (dataset_dir / split / 'labels' / f"{prefix}_{img_path.stem}.txt").write_text('\n'.join(new_lines))

                image_hashes[img_hash] = True
                stats['images'] += 1

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


def upload_to_s3(job_dir, bucket, job_id):
    """Upload job to S3 (including train.py)."""
    s3 = boto3.client('s3')

    print(f"\nUploading to s3://{bucket}/jobs/{job_id}/")

    count = 0
    for path in job_dir.rglob('*'):
        if path.is_file():
            key = f"jobs/{job_id}/{path.relative_to(job_dir)}"
            s3.upload_file(str(path), bucket, key)
            count += 1

    # Upload train.py with job (self-contained)
    train_py = Path(__file__).parent / 'train.py'
    if train_py.exists():
        s3.upload_file(str(train_py), bucket, f"jobs/{job_id}/train.py")
        count += 1

    print(f"  Uploaded {count} files")


def create_spot_request(job_id, instance_type, bucket, infra):
    """Create a persistent spot request for this job."""
    ec2 = boto3.client('ec2')

    # Check if already running
    response = ec2.describe_spot_instance_requests(
        Filters=[
            {'Name': 'state', 'Values': ['open', 'active']},
            {'Name': 'tag:JobId', 'Values': [job_id]}
        ]
    )
    if response['SpotInstanceRequests']:
        print(f"\nSpot request already exists for {job_id}")
        return

    user_data = f"""#!/bin/bash
export JOB_ID="{job_id}"
export S3_BUCKET="{bucket}"
export EFS_ID="{infra['efs_id']}"

# Mount EFS
mkdir -p /mnt/efs
mount -t efs {infra['efs_id']}:/ /mnt/efs

# Install dependencies
pip install ultralytics boto3 pyyaml requests

# Download and run trainer
aws s3 cp s3://{bucket}/jobs/{job_id}/train.py /home/ubuntu/train.py
cd /home/ubuntu
python train.py
"""

    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    ec2.request_spot_instances(
        InstanceCount=1,
        Type='persistent',
        LaunchSpecification={
            'ImageId': infra['ami_id'],
            'InstanceType': instance_type,
            'UserData': user_data_b64,
            'SubnetId': infra['subnet_id'],
            'SecurityGroupIds': [infra['security_group_id']],
            'IamInstanceProfile': {'Name': infra['iam_instance_profile']},
            'BlockDeviceMappings': [
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': 100,
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }
            ]
        },
        TagSpecifications=[
            {
                'ResourceType': 'spot-instances-request',
                'Tags': [
                    {'Key': 'Name', 'Value': f'trainer-{job_id}'},
                    {'Key': 'JobId', 'Value': job_id}
                ]
            }
        ]
    )

    print(f"\nCreated spot request for {job_id} on {instance_type}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
