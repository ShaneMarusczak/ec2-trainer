#!/bin/bash
# EC2 Bootstrap Script (User Data)
# This script is embedded in the spot request by Lambda

set -e

# These are set by Lambda in user data
# JOB_ID, S3_BUCKET, SNS_TOPIC_ARN, EFS_ID

echo "=== Starting bootstrap for job: $JOB_ID ==="

# Install EFS mount helper if needed
if ! command -v mount.efs &> /dev/null; then
    apt-get update -y
    apt-get install -y amazon-efs-utils nfs-common
fi

# Mount EFS
mkdir -p /mnt/efs
if ! mountpoint -q /mnt/efs; then
    mount -t efs $EFS_ID:/ /mnt/efs
fi
echo "EFS mounted at /mnt/efs"

# Install Python dependencies
pip install --upgrade pip
pip install ultralytics boto3 pyyaml requests

# Download trainer script
aws s3 cp s3://$S3_BUCKET/trainer/train.py /home/ubuntu/train.py

# Export environment for train.py
export JOB_ID
export S3_BUCKET
export SNS_TOPIC_ARN
export EFS_ID

# Run trainer
cd /home/ubuntu
python train.py

echo "=== Bootstrap complete ==="
