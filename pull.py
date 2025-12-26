#!/usr/bin/env python3
"""
Check job status and pull trained weights.

Usage:
    python pull.py              # show status, optionally sync weights
    python pull.py ~/models     # sync to custom directory
"""

import sys
import boto3
import yaml
from pathlib import Path

CONFIG_FILE = Path.home() / '.ec2-trainer.yaml'


def main():
    local_dir = Path(sys.argv[1] if len(sys.argv) > 1 else './weights')

    # Load config
    if not CONFIG_FILE.exists():
        print("No config found. Run prep.py first.")
        return

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    bucket = config.get('bucket')
    if not bucket:
        print("No bucket configured. Run prep.py first.")
        return

    ec2 = boto3.client('ec2')
    s3 = boto3.client('s3')

    # Get active spot requests
    print(f"\nBucket: {bucket}\n")

    response = ec2.describe_spot_instance_requests(
        Filters=[
            {'Name': 'state', 'Values': ['open', 'active']},
            {'Name': 'tag-key', 'Values': ['JobId']}
        ]
    )

    active_jobs = {}
    for req in response['SpotInstanceRequests']:
        job_id = next((t['Value'] for t in req.get('Tags', []) if t['Key'] == 'JobId'), None)
        if job_id:
            status = "running" if req.get('InstanceId') else req.get('Status', {}).get('Code', 'pending')
            instance_type = req['LaunchSpecification']['InstanceType']
            active_jobs[job_id] = f"{status} ({instance_type})"

    if active_jobs:
        print("Active:")
        for job_id, status in active_jobs.items():
            print(f"  {job_id}: {status}")

    # Get completed jobs (have weights in S3)
    completed = set()
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='weights/', Delimiter='/'):
        for prefix in page.get('CommonPrefixes', []):
            job_id = prefix['Prefix'].split('/')[1]
            completed.add(job_id)

    if completed:
        print(f"\nCompleted ({len(completed)}):")
        for job_id in sorted(completed):
            print(f"  {job_id}")

    if not active_jobs and not completed:
        print("No jobs found.")
        return

    if not completed:
        print("\nNo weights to pull yet.")
        return

    # Offer to pull
    choice = input("\nPull weights? [Y/n]: ").strip().lower()
    if choice not in ['', 'y']:
        return

    # Pull weights
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_files = {}
    for page in paginator.paginate(Bucket=bucket, Prefix='weights/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):
                remote_files[key] = obj['Size']

    downloaded = 0
    skipped = 0

    for key, size in remote_files.items():
        relative = key[len('weights/'):]
        local_path = local_dir / relative

        if local_path.exists() and local_path.stat().st_size == size:
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {relative}")
        s3.download_file(bucket, key, str(local_path))
        downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} up to date")
    print(f"Weights: {local_dir.resolve()}")


if __name__ == '__main__':
    main()
