#!/usr/bin/env python3
"""
EC2 YOLO Training - Dataset prep and job launcher.

Usage:
    python prep.py
    python prep.py --reset-roboflow-key
"""

import argparse
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
    parser = argparse.ArgumentParser(description='EC2 YOLO Training - Dataset prep and job launcher')
    parser.add_argument('--reset-roboflow-key', action='store_true',
                        help='Reset your Roboflow API key')
    parser.add_argument('--set-ntfy', metavar='TOPIC',
                        help='Set ntfy.sh topic for notifications')
    parser.add_argument('--set-key', metavar='KEY_NAME',
                        help='Set EC2 key pair name for SSH access')
    args = parser.parse_args()

    print("=" * 60)
    print("  EC2 YOLO Training")
    print("=" * 60)

    # Load or create infrastructure config
    infra = load_infra_config()

    # Handle Roboflow key reset
    if args.reset_roboflow_key:
        reset_roboflow_key(infra)
        return

    # Handle ntfy topic setting
    if args.set_ntfy:
        infra['ntfy_topic'] = args.set_ntfy
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(infra, f)
        print(f"\nntfy topic set to: {args.set_ntfy}")
        print(f"You'll receive notifications at: https://ntfy.sh/{args.set_ntfy}")
        return

    # Handle key pair setting
    if args.set_key:
        infra['key_name'] = args.set_key
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(infra, f)
        print(f"\nSSH key pair set to: {args.set_key}")
        return

    # Show existing jobs if we have a saved bucket
    saved_bucket = infra.get('bucket')
    if saved_bucket:
        list_jobs(saved_bucket)

    # Step 1: Job ID
    print("\nJob ID (e.g., spaghetti-v1):")
    job_id = input("> ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("> ").strip()

    # Check if job exists in S3 (if we have a bucket)
    if saved_bucket and job_exists(saved_bucket, job_id):
        print(f"\nJob '{job_id}' exists in S3.")
        print("  [L]aunch instance")
        print("  [O]verwrite (re-upload)")
        print("  [C]ancel")
        choice = input("\n> ").strip().lower()

        if choice == 'l':
            # Just launch - get instance type and go
            print("\n  Instance (common: g5.xlarge, g5.2xlarge, g4dn.xlarge):")
            instance_type = input("  [g5.xlarge]: ").strip() or "g5.xlarge"
            create_spot_request(job_id, instance_type, saved_bucket, infra)
            print("\nTraining started!")
            return
        elif choice != 'o':
            print("Cancelled.")
            return
        # Otherwise continue to overwrite

    # Step 2: Collect datasets
    print("\nDatasets (Enter when done):")
    print("  - Local path: ~/datasets/my-dataset")
    print("  - Roboflow:   rf:workspace/project/version")
    print("  - Previous:   job:<job_id>  (reuse merged dataset)")
    print("  - Reset key:  rf:reset")
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

        # Check if Roboflow or previous job
        if user_input.lower() == 'rf:reset':
            reset_roboflow_key(infra)
            continue
        elif user_input.startswith('rf:'):
            path = download_roboflow(user_input, infra)
            if not path:
                continue
        elif user_input.startswith('job:'):
            # Reuse merged dataset from previous job
            prev_job_id = user_input[4:].strip()
            path = Path('./jobs') / prev_job_id / 'dataset'
            if not path.exists():
                print(f"  Not found: {path}")
                print(f"  (Looking for ./jobs/{prev_job_id}/dataset)")
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

    # Step 3: Analyze datasets and select classes
    dataset_configs, class_counts = analyze_datasets(datasets)
    if not class_counts:
        print("\nNo classes found in datasets!")
        return

    final_classes = select_classes(class_counts)
    if not final_classes:
        print("Cancelled.")
        return

    # Step 4: Training config
    training_config = get_training_config()

    # Step 5: Pick bucket
    bucket = pick_bucket(infra)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    if len(datasets) == 1:
        print(f"  Dataset:  {datasets[0].name}")
    else:
        print(f"  Datasets: {len(datasets)} (will merge)")
        for ds in datasets:
            print(f"            - {ds.name}")
    print(f"  Classes:  {final_classes}")
    print(f"  Job ID:   {job_id}")
    print(f"  Model:    {training_config['model']}")
    print(f"  Instance: {training_config['instance_type']}")
    print(f"  Epochs:   {training_config['epochs']}")
    print(f"  Bucket:   {bucket}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    # Step 7: Merge datasets (non-interactive)
    job_dir = merge_datasets(dataset_configs, final_classes, job_id)
    if not job_dir:
        print("Failed to merge datasets.")
        return

    # Write training config
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)

    # Step 8: Upload to S3
    upload_to_s3(job_dir, bucket, job_id)

    # Step 9: Create spot request
    create_spot_request(job_id, training_config['instance_type'], bucket, infra)

    print("\nTraining started!")


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
        'ami_id': input("AMI ID [ami-0ce8c5eb104aa745d]: ").strip() or 'ami-0ce8c5eb104aa745d',
        'key_name': input("EC2 Key Pair name (for SSH, optional): ").strip() or None,
        'ntfy_topic': input("ntfy.sh topic (for notifications, optional): ").strip() or None,
    }

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)
    print(f"\nSaved to {CONFIG_FILE}")

    return config


def reset_roboflow_key(infra):
    """Reset the Roboflow API key."""
    current_key = infra.get('roboflow_api_key')
    if current_key:
        masked = current_key[:4] + '...' + current_key[-4:] if len(current_key) > 8 else '****'
        print(f"\nCurrent Roboflow API key: {masked}")
    else:
        print("\nNo Roboflow API key currently configured.")

    print("\nEnter new Roboflow API key (from roboflow.com/settings):")
    print("  (Press Enter to cancel)")
    new_key = input("> ").strip()

    if not new_key:
        print("Cancelled.")
        return

    infra['roboflow_api_key'] = new_key
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(infra, f)
    print(f"\nRoboflow API key updated in {CONFIG_FILE}")


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

    # Check if already downloaded
    dest_path = DATASETS_DIR / f"{workspace}_{project}_v{version}"
    if dest_path.exists() and (dest_path / 'data.yaml').exists():
        print(f"  Already downloaded: {dest_path}")
        return dest_path

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
        error_msg = str(e).lower()
        # Check for authentication-related errors
        if any(keyword in error_msg for keyword in ['unauthorized', 'invalid api', 'authentication', '401', 'forbidden', '403']):
            print(f"  Authentication failed: {e}")
            print("\n  Your API key may be invalid or expired.")
            print("  Would you like to enter a new API key? [Y/n]:")
            if input("  > ").strip().lower() in ['', 'y', 'yes']:
                print("\n  Enter new Roboflow API key (from roboflow.com/settings):")
                new_key = input("  > ").strip()
                if new_key:
                    infra['roboflow_api_key'] = new_key
                    with open(CONFIG_FILE, 'w') as f:
                        yaml.dump(infra, f)
                    print("  API key updated. Please try the download again.")
        else:
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


def list_jobs(bucket):
    """List existing jobs and their status."""
    s3 = boto3.client('s3')
    ec2 = boto3.client('ec2')

    # Get jobs from S3
    jobs = {}
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix='jobs/', Delimiter='/'):
            for prefix in page.get('CommonPrefixes', []):
                job_id = prefix['Prefix'].split('/')[1]
                jobs[job_id] = {'status': 'pending'}
    except Exception:
        return

    if not jobs:
        return

    # Check which have weights (complete)
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix='weights/', Delimiter='/'):
            for prefix in page.get('CommonPrefixes', []):
                job_id = prefix['Prefix'].split('/')[1]
                if job_id in jobs:
                    jobs[job_id]['status'] = 'complete'
    except Exception:
        pass

    # Check active spot requests
    try:
        response = ec2.describe_spot_instance_requests(
            Filters=[
                {'Name': 'state', 'Values': ['open', 'active']},
                {'Name': 'tag-key', 'Values': ['JobId']}
            ]
        )
        for req in response['SpotInstanceRequests']:
            job_id = next((t['Value'] for t in req.get('Tags', []) if t['Key'] == 'JobId'), None)
            if job_id and job_id in jobs:
                if req.get('InstanceId'):
                    jobs[job_id]['status'] = 'running'
                    jobs[job_id]['instance'] = req['LaunchSpecification']['InstanceType']
                else:
                    jobs[job_id]['status'] = 'starting'
    except Exception:
        pass

    # Display
    print("\nExisting jobs:")
    for job_id in sorted(jobs.keys()):
        info = jobs[job_id]
        status = info['status']
        if status == 'complete':
            print(f"  {job_id}: complete")
        elif status == 'running':
            print(f"  {job_id}: running ({info.get('instance', '?')})")
        elif status == 'starting':
            print(f"  {job_id}: starting...")
        else:
            print(f"  {job_id}: pending")


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


def analyze_datasets(dataset_paths):
    """Analyze datasets: normalize classes and count images per class.

    Returns (dataset_configs, class_counts) where:
    - dataset_configs: list of dicts with path, original_classes, normalized_classes
    - class_counts: Counter of normalized class -> image count
    """
    print(f"\nAnalyzing {len(dataset_paths)} dataset(s)...")

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
        print(f"  {ds_path.name}: {classes} -> {[n for n in normalized if n]}")

    if unknown_classes:
        print(f"\n  Unknown classes (kept as-is): {unknown_classes}")

    # Count images per normalized class
    class_counts = Counter()

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
                    class_counts[cls] += 1

    return dataset_configs, class_counts


def select_classes(class_counts):
    """Interactive class selection. Returns sorted list of selected classes or None."""
    print("\nClass distribution:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    for i, (cls, count) in enumerate(sorted_classes, 1):
        print(f"  {i}. {cls}: {count} images")

    print("\nSelect classes to keep:")
    print("  - Enter numbers (e.g., 1,2,3 or 1-3)")
    print("  - 'all' to keep all classes")
    print("  - Enter for default (classes with 50+ images)")

    selection = input("\n> ").strip().lower()

    if selection == 'all':
        final_classes = [cls for cls, _ in sorted_classes]
    elif selection == '':
        # Default: filter by MIN_IMAGES_PER_CLASS
        final_classes = [cls for cls, count in sorted_classes if count >= MIN_IMAGES_PER_CLASS]
        if not final_classes:
            print("\n  No classes have 50+ images. Keeping all.")
            final_classes = [cls for cls, _ in sorted_classes]
    else:
        # Parse selection (e.g., "1,2,3" or "1-3" or "1,3-5")
        selected_indices = set()
        for part in selection.replace(' ', '').split(','):
            if '-' in part:
                start, end = part.split('-', 1)
                try:
                    for i in range(int(start), int(end) + 1):
                        selected_indices.add(i)
                except ValueError:
                    pass
            else:
                try:
                    selected_indices.add(int(part))
                except ValueError:
                    pass

        final_classes = []
        for i, (cls, _) in enumerate(sorted_classes, 1):
            if i in selected_indices:
                final_classes.append(cls)

    if not final_classes:
        return None

    return sorted(final_classes)


def merge_datasets(dataset_configs, final_classes, job_id):
    """Merge datasets with deduplication. Non-interactive."""
    print("\nMerging datasets...")

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

                # Deduplicate by hash (not for security, just dedup)
                img_hash = hashlib.md5(img_path.read_bytes(), usedforsecurity=False).hexdigest()
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

    print(f"  Images: {stats['images']}")
    print(f"  Annotations: {stats['annotations']}")
    print(f"  Duplicates removed: {stats['duplicates']}")
    print(f"  Annotations dropped: {stats['dropped']}")

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

    ntfy_topic = infra.get('ntfy_topic', '')
    user_data = f"""#!/bin/bash
set -e  # Exit on error

export JOB_ID="{job_id}"
export S3_BUCKET="{bucket}"
export EFS_ID="{infra['efs_id']}"
export NTFY_TOPIC="{ntfy_topic}"

# ntfy notification helper
ntfy() {{
    [ -n "$NTFY_TOPIC" ] && curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
}}

# Log everything to S3 for debugging
exec > >(tee /var/log/user-data.log) 2>&1
trap 'aws s3 cp /var/log/user-data.log s3://{bucket}/logs/{job_id}.log || true' EXIT

echo "Starting job {job_id} at $(date)"
ntfy "🚀 [{job_id}] EC2 spot fulfilled, bootstrapping..."

# Install EFS utils if needed
if ! command -v mount.efs &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq amazon-efs-utils
fi

# Mount EFS with retry
mkdir -p /mnt/efs
for i in 1 2 3 4 5; do
    mount -t efs {infra['efs_id']}:/ /mnt/efs && break
    echo "EFS mount attempt $i failed, retrying in 10s..."
    sleep 10
done

# Verify mount
if ! mountpoint -q /mnt/efs; then
    echo "ERROR: EFS mount failed after 5 attempts"
    ntfy "❌ [{job_id}] EFS mount failed!"
    exit 1
fi
ntfy "✓ [{job_id}] EFS mounted"

# Activate PyTorch env and install deps
source /opt/conda/bin/activate pytorch
pip install -q ultralytics boto3 pyyaml requests
ntfy "✓ [{job_id}] Dependencies installed"

# Download and run trainer
aws s3 cp s3://{bucket}/jobs/{job_id}/train.py /home/ubuntu/train.py
cd /home/ubuntu
python train.py
"""

    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    # Build launch specification
    launch_spec = {
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
        }

    # Add SSH key if configured
    if infra.get('key_name'):
        launch_spec['KeyName'] = infra['key_name']

    ec2.request_spot_instances(
        InstanceCount=1,
        Type='persistent',
        LaunchSpecification=launch_spec,
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
