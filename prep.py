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
    existing_jobs = []
    if saved_bucket:
        existing_jobs = list_jobs(saved_bucket) or []

    # Step 1: Job ID - show menu if jobs exist
    if existing_jobs:
        print("\nExisting jobs:")
        for i, (jid, info) in enumerate(existing_jobs, 1):
            status = info['status']
            if status == 'complete':
                status_str = 'complete'
            elif status == 'running':
                status_str = f"running ({info.get('instance', '?')})"
            elif status == 'starting':
                status_str = 'starting...'
            else:
                status_str = 'pending'
            print(f"  {i}. {jid}: {status_str}")
        print("\nSelect job number or enter new name:")
    else:
        print("\nJob ID:")

    job_id = input("> ").strip()
    while not job_id or ' ' in job_id:
        job_id = input("> ").strip()

    # If they entered a number, map to existing job
    if job_id.isdigit():
        idx = int(job_id) - 1
        if 0 <= idx < len(existing_jobs):
            job_id = existing_jobs[idx][0]
        else:
            print(f"Invalid selection. Enter 1-{len(existing_jobs)} or a new name.")
            return

    # Check if job exists in S3 (if we have a bucket)
    overwriting_job = None
    if saved_bucket and job_exists(saved_bucket, job_id):
        print(f"\nJob '{job_id}' exists in S3.")
        print("  [L]aunch instance")
        print("  [O]verwrite (re-upload)")
        print("  [C]ancel")
        choice = input("\n> ").strip().lower()

        if choice == 'l':
            # Just launch - get instance type and launch options
            print("\n  Instance (common: g5.xlarge, g5.2xlarge, g4dn.xlarge):")
            instance_type = input("  [g5.xlarge]: ").strip() or "g5.xlarge"
            print("\n  Launch type:")
            print("    [S]pot - may wait for capacity")
            print("    [O]n-demand - starts immediately")
            launch_type = input("  [S]: ").strip().lower() or "s"
            use_spot = launch_type != 'o'
            launch_instance(job_id, instance_type, saved_bucket, infra, use_spot)
            return
        elif choice != 'o':
            print("Cancelled.")
            return
        # Otherwise continue to overwrite
        overwriting_job = job_id

    # Step 2: Collect datasets
    datasets = []

    # Offer to reuse dataset if overwriting
    reusing_merged = False
    if overwriting_job:
        print(f"\nReuse dataset from '{overwriting_job}'? [Y/n]")
        if input("> ").strip().lower() != 'n':
            reuse_path = Path('./jobs') / overwriting_job / 'dataset'
            if reuse_path.exists():
                datasets.append(reuse_path)
                reusing_merged = True
                print(f"  Using dataset from {overwriting_job}")
            else:
                print(f"  Dataset not found at {reuse_path}, enter manually")

    if not datasets:
        print("\nDatasets (Enter when done):")
        print("  - Local path: ~/datasets/my-dataset")
        print("  - Roboflow:   rf:workspace/project/version")
        print("  - Previous:   job:<job_id>  (reuse merged dataset)")
        print("  - Reset key:  rf:reset")
        print()

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

    # If reusing merged dataset, skip analysis/selection/merge
    if reusing_merged:
        # Read classes from the merged dataset's data.yaml
        with open(datasets[0] / 'data.yaml') as f:
            config = yaml.safe_load(f)
        final_classes = config.get('names', [])
        if isinstance(final_classes, dict):
            final_classes = list(final_classes.values())
        job_dir = Path('./jobs') / job_id
        dataset_configs = None  # Not needed
    else:
        # Step 3: Analyze datasets and select classes
        dataset_configs, class_counts = analyze_datasets(datasets)
        if not class_counts:
            print("\nNo classes found in datasets!")
            return

        final_classes = select_classes(class_counts)
        if not final_classes:
            print("Cancelled.")
            return
        job_dir = None  # Will be created by merge

    # Step 4: Training config
    training_config = get_training_config()

    # Step 5: Pick bucket
    bucket = pick_bucket(infra)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    if reusing_merged:
        print(f"  Dataset:  {datasets[0].name} (reusing)")
    elif len(datasets) == 1:
        print(f"  Dataset:  {datasets[0].name}")
    else:
        print(f"  Datasets: {len(datasets)} (will merge)")
        for ds in datasets:
            print(f"            - {ds.name}")
    print(f"  Classes:  {final_classes}")
    print(f"  Job ID:   {job_id}")
    print(f"  Model:    {training_config['model']}")
    print(f"  Instance: {training_config['instance_type']}")
    print(f"  Launch:   {'spot' if training_config.get('use_spot', True) else 'on-demand'}")
    print(f"  Epochs:   {training_config['epochs']}")
    print(f"  Bucket:   {bucket}")

    if input("\nProceed? [Y/n]: ").strip().lower() not in ['', 'y']:
        print("Cancelled.")
        return

    # Step 7: Merge datasets (skip if reusing)
    if not reusing_merged:
        job_dir = merge_datasets(dataset_configs, final_classes, job_id)
        if not job_dir:
            print("Failed to merge datasets.")
            return

    # Write training config (exclude launch-only keys)
    yolo_config = {k: v for k, v in training_config.items() if k not in ('use_spot', 'instance_type')}
    with open(job_dir / 'config.yaml', 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)

    # Step 8: Upload to S3
    upload_to_s3(job_dir, bucket, job_id)

    # Step 9: Launch instance
    launch_instance(
        job_id,
        training_config['instance_type'],
        bucket,
        infra,
        training_config.get('use_spot', True)
    )


def load_infra_config():
    """Load or create infrastructure config, prompting for any missing fields."""
    # Define all fields: (key, prompt, required, default)
    fields = [
        ('subnet_id', 'Subnet ID (subnet-xxxxx)', True, None),
        ('security_group_id', 'Security Group ID (sg-xxxxx)', True, None),
        ('iam_instance_profile', 'IAM Instance Profile name', True, None),
        ('ami_id', 'AMI ID', False, 'ami-03d235ac935098e03'),
        ('key_name', 'EC2 Key Pair name (for SSH, optional)', False, None),
        ('ntfy_topic', 'ntfy.sh topic (for notifications, optional)', False, None),
    ]

    # Load existing config or start fresh
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f) or {}
        print(f"\nLoaded config from {CONFIG_FILE}")

        # Show current config and offer to update
        print("\nCurrent config:")
        for i, (key, prompt, req, default) in enumerate(fields, 1):
            val = config.get(key) or '(not set)'
            print(f"  {i}. {key}: {val}")

        print("\nUpdate config sections? (enter numbers like 1,3,5 or 'all', or press Enter to skip)")
        choice = input("> ").strip().lower()

        if choice:
            # Parse which fields to update
            if choice == 'all':
                to_update = set(range(len(fields)))
            else:
                to_update = set()
                for part in choice.replace(' ', '').split(','):
                    if '-' in part:
                        start, end = part.split('-')
                        to_update.update(range(int(start) - 1, int(end)))
                    elif part.isdigit():
                        to_update.add(int(part) - 1)

            # Prompt for selected fields
            for i in sorted(to_update):
                if 0 <= i < len(fields):
                    key, prompt, required, default = fields[i]
                    current = config.get(key)
                    if current:
                        prompt = f"{prompt} [{current}]"
                    elif default:
                        prompt = f"{prompt} [{default}]"
                    value = input(f"{prompt}: ").strip()
                    if value or not current:
                        config[key] = value or current or default

            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(config, f)
            print(f"\nSaved to {CONFIG_FILE}")
    else:
        config = {}
        print("\nFirst run - need AWS infrastructure config.")
        print("(This will be saved to ~/.ec2-trainer.yaml)\n")

    # Prompt for any missing fields
    updated = False
    for key, prompt, required, default in fields:
        current = config.get(key)
        if current is None:
            if default:
                prompt = f"{prompt} [{default}]"
            value = input(f"{prompt}: ").strip()
            if required:
                while not value and not default:
                    value = input(f"{prompt}: ").strip()
            config[key] = value or default
            updated = True

    # Save if we added anything
    if updated:
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

    # Return sorted list for menu display
    return [(job_id, jobs[job_id]) for job_id in sorted(jobs.keys())]


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

    print("\n  Launch type:")
    print("    [S]pot - may wait for capacity")
    print("    [O]n-demand - starts immediately")
    launch_type = input("  [S]: ").strip().lower() or "s"
    use_spot = launch_type != 'o'

    epochs = int(input("\n  Epochs [120]: ").strip() or "120")
    batch = int(input("  Batch [16]: ").strip() or "16")

    config = {
        'model': model,
        'instance_type': instance_type,
        'use_spot': use_spot,
        'epochs': epochs,
        'batch': batch,
    }

    # Advanced options
    print("\n  Advanced options? [y/N]: ", end="")
    if input().strip().lower() == 'y':
        advanced = get_advanced_config()
        config.update(advanced)

    return config


# Advanced training parameters with defaults and descriptions
ADVANCED_PARAMS = [
    ('lr0', 0.001, 'Initial learning rate'),
    ('imgsz', 640, 'Image size'),
    ('patience', 20, 'Early stopping patience (epochs)'),
    ('optimizer', 'AdamW', 'Optimizer (SGD, Adam, AdamW)'),
    ('warmup_epochs', 5, 'Warmup epochs'),
    ('dropout', 0.1, 'Dropout rate'),
    ('mosaic', 1.0, 'Mosaic augmentation (0-1)'),
    ('mixup', 0.15, 'Mixup augmentation (0-1)'),
    ('degrees', 15, 'Rotation degrees'),
    ('scale', 0.5, 'Scale augmentation'),
    ('close_mosaic', 10, 'Disable mosaic last N epochs'),
]


def get_advanced_config():
    """Prompt for advanced training parameters."""
    print("\n  Advanced parameters (press Enter to keep default):\n")

    # Show all params with numbers
    for i, (name, default, desc) in enumerate(ADVANCED_PARAMS, 1):
        print(f"    {i:2}. {name}: {default} ({desc})")

    print("\n  Enter numbers to modify (e.g., '1 3 5' or 'all'), or Enter to skip:")
    selection = input("  > ").strip().lower()

    if not selection:
        return {}

    # Parse selection
    if selection == 'all':
        indices = list(range(len(ADVANCED_PARAMS)))
    else:
        try:
            indices = [int(x) - 1 for x in selection.split() if x.isdigit()]
            indices = [i for i in indices if 0 <= i < len(ADVANCED_PARAMS)]
        except ValueError:
            return {}

    if not indices:
        return {}

    # Prompt for selected params
    config = {}
    print()
    for idx in indices:
        name, default, desc = ADVANCED_PARAMS[idx]
        value = input(f"    {name} [{default}]: ").strip()
        if value:
            # Parse value based on type of default
            if isinstance(default, float):
                config[name] = float(value)
            elif isinstance(default, int):
                config[name] = int(value)
            else:
                config[name] = value

    return config


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


def launch_instance(job_id, instance_type, bucket, infra, use_spot=True):
    """Launch a spot or on-demand instance for this job."""
    ec2 = boto3.client('ec2')

    # Check if already running (spot request)
    if use_spot:
        response = ec2.describe_spot_instance_requests(
            Filters=[
                {'Name': 'state', 'Values': ['open', 'active']},
                {'Name': 'tag:JobId', 'Values': [job_id]}
            ]
        )
        if response['SpotInstanceRequests']:
            print(f"\nSpot request already exists for {job_id}")
            return

    # Check if already running (on-demand instance)
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'instance-state-name', 'Values': ['pending', 'running']},
            {'Name': 'tag:JobId', 'Values': [job_id]}
        ]
    )
    for reservation in response['Reservations']:
        if reservation['Instances']:
            print(f"\nInstance already running for {job_id}")
            return

    launch_type_str = "spot" if use_spot else "on-demand"
    ntfy_topic = infra.get('ntfy_topic', '')
    user_data = f"""#!/bin/bash
set -e  # Exit on error

export JOB_ID="{job_id}"
export S3_BUCKET="{bucket}"
export NTFY_TOPIC="{ntfy_topic}"

# ntfy notification helper
ntfy() {{
    [ -n "$NTFY_TOPIC" ] && curl -s -d "$1" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
}}

# Log everything to S3 for debugging
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting job {job_id} at $(date)"
ntfy "🚀 [{job_id}] Instance started ({launch_type_str}), bootstrapping..."

# Get region from instance metadata
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

# Safety net: track if we exit cleanly
CLEAN_EXIT=false

# Master trap - runs on ANY exit, catches everything
trap '
    aws s3 cp /var/log/user-data.log s3://{bucket}/logs/{job_id}.log || true
    if [ "$CLEAN_EXIT" != "true" ]; then
        ntfy "💀 [{job_id}] Unexpected exit - self-terminating"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" || true
    fi
' EXIT

# Max runtime failsafe - 12 hours absolute limit
(sleep 43200 && ntfy "⏰ [{job_id}] Max runtime (12h) reached - terminating" && \
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION") &

# Self-terminate on fatal error
terminate() {{
    ntfy "💀 [{job_id}] Fatal error: $1"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" || true
    exit 1
}}

# Activate PyTorch env
if [ -f /opt/pytorch/bin/activate ]; then
    source /opt/pytorch/bin/activate
elif [ -f /opt/conda/bin/activate ]; then
    source /opt/conda/bin/activate pytorch
else
    terminate "No PyTorch environment found"
fi

pip install -q ultralytics boto3 pyyaml requests || terminate "pip install failed"
ntfy "✓ [{job_id}] Dependencies ready"

# Download and run trainer
aws s3 cp s3://{bucket}/jobs/{job_id}/train.py /home/ubuntu/train.py || terminate "Failed to download train.py"
cd /home/ubuntu
export AWS_DEFAULT_REGION=$REGION
python train.py || terminate "Training failed"

# Success
CLEAN_EXIT=true
ntfy "✅ [{job_id}] Complete"
"""

    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    # Save user_data for debugging
    debug_script = Path('./jobs') / job_id / 'user_data.sh'
    debug_script.write_text(user_data)
    print(f"\n  Debug: user_data saved to {debug_script}")

    # Common instance config
    instance_config = {
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
        instance_config['KeyName'] = infra['key_name']

    tags = [
        {'Key': 'Name', 'Value': f'trainer-{job_id}'},
        {'Key': 'JobId', 'Value': job_id}
    ]

    if use_spot:
        ec2.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification=instance_config,
            TagSpecifications=[
                {
                    'ResourceType': 'spot-instances-request',
                    'Tags': tags
                }
            ]
        )
        print(f"\nCreated spot request for {job_id} on {instance_type}")
    else:
        ec2.run_instances(
            MinCount=1,
            MaxCount=1,
            **instance_config,
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': tags
                }
            ]
        )
        print(f"\nLaunched on-demand instance for {job_id} on {instance_type}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
