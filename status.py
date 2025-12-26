#!/usr/bin/env python3
"""
Check status of training jobs.

Usage:
    python status.py
"""

import boto3
import yaml
from pathlib import Path

CONFIG_FILE = Path.home() / '.ec2-trainer.yaml'


def main():
    # Load config for bucket
    if not CONFIG_FILE.exists():
        print("No config found. Run prep.py first.")
        return

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    bucket = config.get('bucket')
    if not bucket:
        print("No bucket configured.")
        return

    ec2 = boto3.client('ec2')
    s3 = boto3.client('s3')

    # Get active spot requests
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
            active_jobs[job_id] = {
                'state': req['State'],
                'status': req.get('Status', {}).get('Code', ''),
                'instance': req.get('InstanceId', '-'),
                'type': req['LaunchSpecification']['InstanceType']
            }

    # Get completed jobs (have weights)
    completed = set()
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='weights/', Delimiter='/'):
        for prefix in page.get('CommonPrefixes', []):
            job_id = prefix['Prefix'].split('/')[1]
            completed.add(job_id)

    # Display
    print(f"\nBucket: {bucket}\n")

    if active_jobs:
        print("Active jobs:")
        for job_id, info in active_jobs.items():
            status = "running" if info['instance'] != '-' else info['status']
            print(f"  {job_id}: {status} ({info['type']})")
    else:
        print("No active jobs.")

    if completed:
        print(f"\nCompleted ({len(completed)}):")
        for job_id in sorted(completed):
            marker = " (active)" if job_id in active_jobs else ""
            print(f"  {job_id}{marker}")
    else:
        print("\nNo completed jobs yet.")


if __name__ == '__main__':
    main()
