#!/usr/bin/env python3
"""
One-time AWS infrastructure setup for EC2 YOLO Training.

Creates: S3 bucket, Security Group, IAM role, and saves config.

Usage:
    python setup.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Auto-install dependencies
for pkg, imp in [('boto3', 'boto3'), ('pyyaml', 'yaml')]:
    try:
        __import__(imp)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import boto3
import yaml

CONFIG_FILE = Path.home() / '.ec2-trainer.yaml'


def main():
    print("=" * 60)
    print("  EC2 YOLO Training - AWS Setup")
    print("=" * 60)

    if CONFIG_FILE.exists():
        print(f"\nConfig already exists at {CONFIG_FILE}")
        if input("Overwrite? [y/N]: ").strip().lower() != 'y':
            print("Cancelled.")
            return

    # Get region and VPC
    ec2 = boto3.client('ec2')
    region = ec2.meta.region_name
    print(f"\nRegion: {region}")

    # List VPCs
    vpcs = ec2.describe_vpcs()['Vpcs']
    print("\nVPCs:")
    for i, vpc in enumerate(vpcs, 1):
        name = next((t['Value'] for t in vpc.get('Tags', []) if t['Key'] == 'Name'), '-')
        default = " (default)" if vpc.get('IsDefault') else ""
        print(f"  {i}. {vpc['VpcId']} - {name}{default}")

    choice = input("\nSelect VPC [1]: ").strip() or "1"
    vpc_id = vpcs[int(choice) - 1]['VpcId']

    # List subnets
    subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])['Subnets']
    print("\nSubnets:")
    for i, subnet in enumerate(subnets, 1):
        name = next((t['Value'] for t in subnet.get('Tags', []) if t['Key'] == 'Name'), '-')
        print(f"  {i}. {subnet['SubnetId']} - {subnet['AvailabilityZone']} - {name}")

    choice = input("\nSelect Subnet [1]: ").strip() or "1"
    subnet = subnets[int(choice) - 1]
    subnet_id = subnet['SubnetId']
    az = subnet['AvailabilityZone']

    # Bucket name
    account_id = boto3.client('sts').get_caller_identity()['Account']
    default_bucket = f"yolo-training-{account_id}-{region}"
    bucket_name = input(f"\nS3 Bucket name [{default_bucket}]: ").strip() or default_bucket

    print("\nCreating infrastructure...")

    # 1. S3 Bucket
    print("  Creating S3 bucket...")
    s3 = boto3.client('s3')
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    print(f"    ✓ {bucket_name}")

    # 2. Security Group
    print("  Creating security group...")
    try:
        sg_response = ec2.create_security_group(
            GroupName='yolo-trainer',
            Description='YOLO training instances - outbound only',
            VpcId=vpc_id
        )
        sg_id = sg_response['GroupId']
    except ec2.exceptions.ClientError as e:
        if 'already exists' in str(e):
            sgs = ec2.describe_security_groups(
                Filters=[
                    {'Name': 'group-name', 'Values': ['yolo-trainer']},
                    {'Name': 'vpc-id', 'Values': [vpc_id]}
                ]
            )['SecurityGroups']
            sg_id = sgs[0]['GroupId']
        else:
            raise
    print(f"    ✓ {sg_id}")

    # 3. IAM Role
    print("  Creating IAM role...")
    iam = boto3.client('iam')

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        iam.create_role(
            RoleName='yolo-trainer',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role for YOLO training EC2 instances'
        )
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    # Attach policies
    policies = [
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        'arn:aws:iam::aws:policy/AmazonEC2FullAccess',
    ]
    for policy in policies:
        try:
            iam.attach_role_policy(RoleName='yolo-trainer', PolicyArn=policy)
        except iam.exceptions.ClientError:
            pass

    # Instance profile
    try:
        iam.create_instance_profile(InstanceProfileName='yolo-trainer')
        time.sleep(2)  # Wait for propagation
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName='yolo-trainer',
            RoleName='yolo-trainer'
        )
    except iam.exceptions.LimitExceededException:
        pass  # Already added
    print("    ✓ yolo-trainer")

    # Save config
    config = {
        'bucket': bucket_name,
        'subnet_id': subnet_id,
        'security_group_id': sg_id,
        'iam_instance_profile': 'yolo-trainer',
        'ami_id': 'ami-0ce8c5eb104aa745d',  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
    }

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'=' * 60}")
    print("  Setup Complete!")
    print("=" * 60)
    print(f"\nConfig saved to {CONFIG_FILE}")
    print("\nYou can now run: ./run.sh")


if __name__ == '__main__':
    main()
