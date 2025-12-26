#!/usr/bin/env python3
"""
Pull trained weights from S3.

Usage:
    python pull.py              # Sync all weights to ./weights/
    python pull.py ~/models     # Sync to custom directory
"""

import sys
import boto3
from pathlib import Path

def main():
    local_dir = Path(sys.argv[1] if len(sys.argv) > 1 else './weights')
    local_dir.mkdir(parents=True, exist_ok=True)

    # Pick bucket
    print("S3 bucket:")
    try:
        response = boto3.client('s3').list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        if buckets:
            for i, b in enumerate(buckets, 1):
                print(f"  {i}. {b}")
            choice = input("\n> ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(buckets):
                bucket = buckets[int(choice) - 1]
            else:
                bucket = choice or buckets[0]
        else:
            bucket = input("\n> ").strip()
    except Exception:
        bucket = input("\n> ").strip()

    s3 = boto3.client('s3')

    # List remote weights
    print(f"\nChecking s3://{bucket}/weights/...")
    paginator = s3.get_paginator('list_objects_v2')

    remote_files = {}
    for page in paginator.paginate(Bucket=bucket, Prefix='weights/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            remote_files[key] = obj['Size']

    if not remote_files:
        print("No weights found in S3.")
        return

    # Check local and download missing
    downloaded = 0
    skipped = 0

    for key, size in remote_files.items():
        # weights/job-id/best.pt -> ./weights/job-id/best.pt
        relative = key[len('weights/'):]
        local_path = local_dir / relative

        if local_path.exists() and local_path.stat().st_size == size:
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {relative}...")
        s3.download_file(bucket, key, str(local_path))
        downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} already up to date")
    print(f"Weights in: {local_dir.resolve()}")


if __name__ == '__main__':
    main()
