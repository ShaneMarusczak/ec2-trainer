"""
Lambda handler: S3 trigger → Create spot request for job

Triggered when a config.yaml is uploaded to s3://bucket/jobs/{job_id}/config.yaml
Creates a persistent spot request bound to that specific job.
"""

import boto3
import yaml
import os
from botocore.exceptions import ClientError

ec2 = boto3.client('ec2')
s3 = boto3.client('s3')


def get_required_env(key):
    """Get required environment variable or raise clear error."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# Environment variables (set in Lambda config)
BUCKET = get_required_env('BUCKET')
EFS_ID = get_required_env('EFS_ID')
SNS_TOPIC_ARN = get_required_env('SNS_TOPIC_ARN')
SUBNET_ID = get_required_env('SUBNET_ID')
SECURITY_GROUP_ID = get_required_env('SECURITY_GROUP_ID')
IAM_INSTANCE_PROFILE = get_required_env('IAM_INSTANCE_PROFILE')
AMI_ID = os.environ.get('AMI_ID', 'ami-0c7217cdde317cfec')  # Deep Learning AMI


def handler(event, context):
    """
    S3 trigger handler.
    Expects upload to: s3://bucket/jobs/{job_id}/config.yaml
    """
    
    # Extract job_id from S3 event
    record = event['Records'][0]
    key = record['s3']['object']['key']  # "jobs/chicken-detector/config.yaml"
    
    parts = key.split('/')
    if len(parts) < 3 or parts[0] != 'jobs' or parts[-1] != 'config.yaml':
        print(f"Ignoring non-config upload: {key}")
        return
    
    job_id = parts[1]
    print(f"Job detected: {job_id}")
    
    # Check if already complete
    if job_complete(job_id):
        print(f"Job {job_id} already complete in weights/")
        return
    
    # Check if spot request already exists for this job
    if spot_exists_for_job(job_id):
        print(f"Spot request already exists for {job_id}")
        return
    
    # Read config to get instance type
    config = get_job_config(job_id)
    instance_type = config.get('instance_type', 'g5.xlarge')
    
    # Create spot request
    create_spot_request(job_id, instance_type)
    print(f"Created spot request for {job_id} on {instance_type}")
    
    return {'statusCode': 200, 'body': f'Spot request created for {job_id}'}


def s3_key_exists(key):
    """Check if a key exists in S3."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise  # Re-raise permission errors, throttling, etc.


def job_complete(job_id):
    """Check if weights already exist for this job."""
    return s3_key_exists(f"weights/{job_id}/best.pt")


def spot_exists_for_job(job_id):
    """Check if there's an active/open spot request for this job."""
    response = ec2.describe_spot_instance_requests(
        Filters=[
            {'Name': 'state', 'Values': ['open', 'active']},
            {'Name': 'tag:JobId', 'Values': [job_id]}
        ]
    )
    return len(response['SpotInstanceRequests']) > 0


def get_job_config(job_id):
    """Read config.yaml from S3."""
    response = s3.get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/config.yaml")
    return yaml.safe_load(response['Body'].read())


def create_spot_request(job_id, instance_type):
    """Create a persistent spot request for this job."""
    
    user_data = f"""#!/bin/bash
export JOB_ID="{job_id}"
export S3_BUCKET="{BUCKET}"
export SNS_TOPIC_ARN="{SNS_TOPIC_ARN}"
export EFS_ID="{EFS_ID}"

# Mount EFS
mkdir -p /mnt/efs
mount -t efs {EFS_ID}:/ /mnt/efs

# Install dependencies
pip install ultralytics boto3 pyyaml requests

# Download and run trainer
aws s3 cp s3://{BUCKET}/trainer/train.py /home/ubuntu/train.py
cd /home/ubuntu
python train.py
"""
    
    import base64
    user_data_b64 = base64.b64encode(user_data.encode()).decode()
    
    ec2.request_spot_instances(
        InstanceCount=1,
        Type='persistent',
        LaunchSpecification={
            'ImageId': AMI_ID,
            'InstanceType': instance_type,
            'UserData': user_data_b64,
            'SubnetId': SUBNET_ID,
            'SecurityGroupIds': [SECURITY_GROUP_ID],
            'IamInstanceProfile': {'Name': IAM_INSTANCE_PROFILE},
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
