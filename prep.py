#!/usr/bin/env python3
"""
EC2 YOLO Training - Dataset prep and job launcher.

Usage:
    python prep.py
"""

import base64
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

# Auto-install dependencies
for pkg, imp in [('boto3', 'boto3'), ('pyyaml', 'yaml')]:
    try:
        __import__(imp)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

from collections import Counter

import boto3
import yaml

CONFIG_FILE = Path.home() / '.ec2-trainer.yaml'
DATASETS_DIR = Path('./datasets')
MIN_IMAGES_PER_CLASS = 50

# Class unification mapping - maps various names to canonical form (None = drop)
CLASS_UNIFICATION = {
    # Spaghetti variants
    'spaghetti': 'spaghetti', 'Spaghetti': 'spaghetti', 'spagatti': 'spaghetti',
    'Spagatti': 'spaghetti', 'spahgetti': 'spaghetti',
    'fail': 'spaghetti', 'failure': 'spaghetti', 'defect': 'spaghetti',
    # Normal/good (drop - we only detect failures)
    'normal': None, 'good': None, 'ok': None, 'OK': None,
    # Other failure types (drop - not spaghetti)
    'bed_adhesion': None, 'bed adhesion': None, 'bed adhesion failure': None,
    'poor initial layer bed adhesion faiure': None, 'adhesion': None,
    'blobs': None, 'Blobs': None, 'blob': None, 'nozzle blob': None,
    'nozzle blob failure': None, 'cracks': None, 'Crack': None, 'crack': None,
    'stringing': None, 'Stringing': None, 'warping': None,
    'under_extrusion': None, 'under-extrusion': None,
    'over_extrusion': None, 'over-extrusion': None,
}


def main():
    print("=" * 60)
    print("  EC2 YOLO Training")
    print("=" * 60)

    # Load or create infrastructure config
    infra = load_infra_config()

    # Collect datasets
    print("\nDatasets to upload (Enter when done):")
    print("  - Local path: ~/datasets/my-dataset")
    print("  - Roboflow:   rf:workspace/project/version")
    print()

    datasets = []
    while True:
        prompt = f"Dataset {len(datasets) + 1}: " if datasets else "Dataset: "
        user_input = input(prompt).strip()

        if not user_input:
            if not datasets:
                print("  Need at least one dataset")
                continue
            break

        # Check if Roboflow
        if user_input.startswith('rf:'):
            path = download_roboflow(user_input, infra)
            if not path:
                continue
        else:
            path = Path(user_input).expanduser()
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

    # Process datasets (unify classes, filter, dedupe)
    job_dir = process_datasets(datasets, job_id)
    if not job_dir:
        print("Failed to process datasets.")
        return

    # Write config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Upload to S3
    upload_to_s3(job_dir, bucket, job_id)

    # Create spot request
    create_spot_request(job_id, config['instance_type'], bucket, infra)

    print("\nTraining started! Run pull.py to check progress.")


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
        'ami_id': input("AMI ID [ami-0a0c8eebcdd6dcbd0]: ").strip() or 'ami-0a0c8eebcdd6dcbd0',
    }

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)
    print(f"\nSaved to {CONFIG_FILE}")

    return config


def download_roboflow(rf_string, infra):
    """Download dataset from Roboflow.

    Format: rf:workspace/project/version or rf:workspace/project (defaults to v1)
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  Installing roboflow...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'roboflow', '-q'])
        from roboflow import Roboflow

    # Parse rf:workspace/project/version
    parts = rf_string[3:].split('/')
    if len(parts) == 2:
        workspace, project = parts
        version = 1
    elif len(parts) == 3:
        workspace, project, version = parts
        version = int(version)
    else:
        print("  Invalid format. Use rf:workspace/project or rf:workspace/project/version")
        return None

    # Get API key
    api_key = infra.get('roboflow_api_key')
    if not api_key:
        print("\n  Roboflow API key (from roboflow.com/settings):")
        api_key = input("  > ").strip()
        if not api_key:
            print("  Cancelled.")
            return None
        # Save for next time
        infra['roboflow_api_key'] = api_key
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(infra, f)

    # Download
    print(f"  Downloading {workspace}/{project} v{version}...")
    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download(
            "yolov8",
            location=str(DATASETS_DIR / f"{workspace}_{project}_v{version}")
        )
        path = DATASETS_DIR / f"{workspace}_{project}_v{version}"
        print(f"  Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  Failed: {e}")
        return None


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


def normalize_class(cls):
    """Normalize a class name using the unification mapping."""
    if cls in CLASS_UNIFICATION:
        return CLASS_UNIFICATION[cls]
    # Try lowercase normalization
    cls_lower = cls.lower().replace(' ', '_').replace('-', '_')
    if cls_lower in ['spaghetti', 'spagatti', 'spahgetti']:
        return 'spaghetti'
    if cls_lower in ['normal', 'good', 'ok']:
        return None
    # Unknown class - keep it but warn
    return cls


def process_datasets(dataset_paths, job_id):
    """Process datasets with class unification, filtering, and deduplication."""
    print(f"\nProcessing {len(dataset_paths)} dataset(s)...")

    # Phase 1: Analyze all datasets and normalize classes
    print("\n  Analyzing classes...")
    dataset_configs = []
    unknown_classes = set()

    for ds_path in dataset_paths:
        with open(ds_path / 'data.yaml') as f:
            config = yaml.safe_load(f)

        classes = config.get('names', [])
        if isinstance(classes, dict):
            classes = list(classes.values())

        normalized = []
        for cls in classes:
            norm = normalize_class(cls)
            normalized.append(norm)
            if norm and cls not in CLASS_UNIFICATION:
                unknown_classes.add(cls)

        dataset_configs.append({
            'path': ds_path,
            'original_classes': classes,
            'normalized_classes': normalized,
        })
        print(f"    {ds_path.name}: {classes} -> {[n for n in normalized if n]}")

    if unknown_classes:
        print(f"\n  Unknown classes (kept as-is): {unknown_classes}")

    # Phase 2: Count images per normalized class
    print("\n  Counting images per class...")
    class_image_counts = Counter()

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        norm_classes = ds_info['normalized_classes']

        for split in ['train', 'valid']:
            lbl_dir = ds_path / split / 'labels'
            if not lbl_dir.exists():
                continue

            for lbl_path in lbl_dir.glob('*.txt'):
                classes_in_image = set()
                for line in lbl_path.read_text().strip().split('\n'):
                    if line.strip():
                        old_id = int(line.split()[0])
                        if old_id < len(norm_classes) and norm_classes[old_id]:
                            classes_in_image.add(norm_classes[old_id])
                for cls in classes_in_image:
                    class_image_counts[cls] += 1

    # Phase 3: Filter classes by minimum image count
    print(f"\n  Class distribution (min {MIN_IMAGES_PER_CLASS} images):")
    final_classes = []
    for cls, count in sorted(class_image_counts.items(), key=lambda x: -x[1]):
        if count >= MIN_IMAGES_PER_CLASS:
            print(f"    {cls}: {count} images [KEEP]")
            final_classes.append(cls)
        else:
            print(f"    {cls}: {count} images [DROP]")

    if not final_classes:
        print("\n  ERROR: No classes have enough images!")
        return None

    final_classes = sorted(final_classes)
    print(f"\n  Final classes: {final_classes}")

    # Phase 4: Merge with deduplication
    print("\n  Merging...")
    job_dir = Path('./jobs') / job_id
    dataset_dir = job_dir / 'dataset'

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ['train', 'valid']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    image_hashes = {}
    stats = {'images': 0, 'annotations': 0, 'duplicates': 0, 'dropped': 0}

    for ds_info in dataset_configs:
        ds_path = ds_info['path']
        norm_classes = ds_info['normalized_classes']

        # Build remap: old_id -> new_id (or None if dropped)
        class_remap = {}
        for i, norm in enumerate(norm_classes):
            if norm and norm in final_classes:
                class_remap[i] = final_classes.index(norm)
            else:
                class_remap[i] = None

        for split in ['train', 'valid']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue

                # Deduplicate by hash
                img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                if img_hash in image_hashes:
                    stats['duplicates'] += 1
                    continue

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                # Remap annotations
                new_lines = []
                for line in lbl_path.read_text().strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split()
                    old_id = int(parts[0])
                    new_id = class_remap.get(old_id)
                    if new_id is not None:
                        parts[0] = str(new_id)
                        new_lines.append(' '.join(parts))
                    else:
                        stats['dropped'] += 1

                if not new_lines:
                    continue

                # Save with dataset prefix
                prefix = ds_path.name.replace(' ', '_').replace('/', '_')
                new_img_name = f"{prefix}_{img_path.name}"
                new_lbl_name = f"{prefix}_{img_path.stem}.txt"

                shutil.copy(img_path, dataset_dir / split / 'images' / new_img_name)
                (dataset_dir / split / 'labels' / new_lbl_name).write_text('\n'.join(new_lines))

                image_hashes[img_hash] = new_img_name
                stats['images'] += 1
                stats['annotations'] += len(new_lines)

    # Write data.yaml
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump({
            'path': '.',
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(final_classes),
            'names': final_classes,
        }, f, default_flow_style=False)

    print(f"\n  Result:")
    print(f"    Images: {stats['images']}")
    print(f"    Annotations: {stats['annotations']}")
    print(f"    Duplicates removed: {stats['duplicates']}")
    print(f"    Annotations dropped: {stats['dropped']}")
    print(f"    Classes: {final_classes}")

    return job_dir


def upload_to_s3(job_dir, bucket, job_id):
    """Upload job to S3 using aws s3 sync (parallel uploads)."""
    print(f"\nUploading to s3://{bucket}/jobs/{job_id}/")

    # Copy train.py into job_dir so it syncs with everything
    train_py = Path(__file__).parent / 'train.py'
    if train_py.exists():
        shutil.copy(train_py, job_dir / 'train.py')

    # Use aws s3 sync for parallel uploads
    cmd = [
        'aws', 's3', 'sync',
        str(job_dir),
        f's3://{bucket}/jobs/{job_id}/',
    ]
    subprocess.run(cmd, check=True)

    # Count files for confirmation
    count = sum(1 for _ in job_dir.rglob('*') if _.is_file())
    print(f"  Synced {count} files")


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

# Activate PyTorch env and install deps
source /opt/conda/bin/activate pytorch
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
