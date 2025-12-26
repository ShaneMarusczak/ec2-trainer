# AWS Setup Guide

One-time setup for the EC2 training pipeline.

---

## Overview

```
You upload job → S3 triggers Lambda → Lambda creates Spot → Spot trains → Weights appear
```

---

## 1. S3 Bucket

Create a bucket for everything:

```bash
aws s3 mb s3://your-training-bucket
```

Structure:
```
s3://your-training-bucket/
  jobs/           # Your input
  weights/        # Your output
  trainer/        # The train.py script
```

Upload the trainer script:
```bash
aws s3 cp trainer/train.py s3://your-training-bucket/trainer/train.py
```

---

## 2. EFS (Elastic File System)

Create EFS for checkpoints that survive spot termination.

### Console:
1. Go to EFS → Create file system
2. Select your VPC
3. Use default settings
4. Note the **File System ID** (e.g., `fs-0123456789abcdef0`)

### Mount targets:
EFS needs a mount target in each AZ where spots might run.
1. Go to EFS → Your filesystem → Network
2. Create mount target for each subnet
3. Use the security group you'll create below

---

## 3. Security Group

Create a security group for EC2 instances:

```bash
aws ec2 create-security-group \
  --group-name training-sg \
  --description "Security group for training instances" \
  --vpc-id vpc-xxx
```

Add rules:
```bash
# SSH (optional, for debugging)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32

# EFS (NFS)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 2049 \
  --source-group sg-xxx

# Outbound (default allows all)
```

---

## 4. IAM Role for EC2

Create a role that EC2 instances will use:

### Trust policy (ec2-trust.json):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Create role:
```bash
aws iam create-role \
  --role-name TrainingInstanceRole \
  --assume-role-policy-document file://ec2-trust.json
```

### Permissions policy (training-permissions.json):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:HeadObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-training-bucket",
        "arn:aws:s3:::your-training-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
      "Resource": "arn:aws:sns:*:*:training-notifications"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeSpotInstanceRequests",
        "ec2:CancelSpotInstanceRequests",
        "ec2:TerminateInstances"
      ],
      "Resource": "*"
    }
  ]
}
```

### Attach policy:
```bash
aws iam put-role-policy \
  --role-name TrainingInstanceRole \
  --policy-name TrainingPermissions \
  --policy-document file://training-permissions.json
```

### Create instance profile:
```bash
aws iam create-instance-profile \
  --instance-profile-name TrainingInstanceProfile

aws iam add-role-to-instance-profile \
  --instance-profile-name TrainingInstanceProfile \
  --role-name TrainingInstanceRole
```

---

## 5. IAM Role for Lambda

Create a role for the Lambda function:

### Trust policy (lambda-trust.json):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Create role:
```bash
aws iam create-role \
  --role-name TrainingLambdaRole \
  --assume-role-policy-document file://lambda-trust.json
```

### Permissions policy (lambda-permissions.json):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:HeadObject"
      ],
      "Resource": "arn:aws:s3:::your-training-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RequestSpotInstances",
        "ec2:DescribeSpotInstanceRequests",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/TrainingInstanceRole"
    }
  ]
}
```

### Attach policy:
```bash
aws iam put-role-policy \
  --role-name TrainingLambdaRole \
  --policy-name LambdaPermissions \
  --policy-document file://lambda-permissions.json
```

---

## 6. SNS Topic (Notifications)

Create a topic for training notifications:

```bash
aws sns create-topic --name training-notifications
```

Subscribe your phone:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:xxx:training-notifications \
  --protocol sms \
  --notification-endpoint +1234567890
```

Or email:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:xxx:training-notifications \
  --protocol email \
  --notification-endpoint you@email.com
```

---

## 7. Lambda Function

### Create deployment package:
```bash
cd lambda
zip handler.zip handler.py
```

### Create function:
```bash
aws lambda create-function \
  --function-name TrainingTrigger \
  --runtime python3.11 \
  --handler handler.handler \
  --role arn:aws:iam::xxx:role/TrainingLambdaRole \
  --zip-file fileb://handler.zip \
  --timeout 30 \
  --environment Variables="{
    BUCKET=your-training-bucket,
    EFS_ID=fs-xxx,
    SNS_TOPIC_ARN=arn:aws:sns:us-east-1:xxx:training-notifications,
    SUBNET_ID=subnet-xxx,
    SECURITY_GROUP_ID=sg-xxx,
    IAM_INSTANCE_PROFILE=TrainingInstanceProfile,
    AMI_ID=ami-xxx
  }"
```

### AMI ID:
Use the AWS Deep Learning AMI for your region:
```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text
```

---

## 8. S3 Event Trigger

Add trigger so Lambda fires on config.yaml upload:

```bash
aws s3api put-bucket-notification-configuration \
  --bucket your-training-bucket \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [
      {
        "LambdaFunctionArn": "arn:aws:lambda:us-east-1:xxx:function:TrainingTrigger",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "prefix", "Value": "jobs/"},
              {"Name": "suffix", "Value": "config.yaml"}
            ]
          }
        }
      }
    ]
  }'
```

### Allow S3 to invoke Lambda:
```bash
aws lambda add-permission \
  --function-name TrainingTrigger \
  --statement-id S3Invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::your-training-bucket
```

---

## Usage

### Add a job:

1. Create job folder:
```
my-job/
  config.yaml
  dataset/
    train/
      images/
      labels/
    valid/
      images/
      labels/
    data.yaml
```

2. Upload:
```bash
aws s3 cp my-job s3://your-training-bucket/jobs/my-job/ --recursive
```

3. Training starts automatically

### config.yaml example:
```yaml
# Model
model: yolo12m.pt
instance_type: g5.xlarge

# Training
epochs: 120
batch: 16
imgsz: 640
patience: 20

# Optimizer
optimizer: AdamW
lr0: 0.001
warmup_epochs: 5

# Loss
box: 0.5
cls: 3.0
dfl: 0.5

# Augmentation
mosaic: 1.0
mixup: 0.15
degrees: 15

# Safety
stall_hours: 2
```

### Check status:
```bash
# Jobs
aws s3 ls s3://your-training-bucket/jobs/

# Completed
aws s3 ls s3://your-training-bucket/weights/
```

### Get weights:
```bash
aws s3 cp s3://your-training-bucket/weights/my-job/best.pt ./
```

### Force retrain:
```bash
aws s3 rm s3://your-training-bucket/weights/my-job/ --recursive
# Then re-upload config.yaml to trigger Lambda
aws s3 cp my-job/config.yaml s3://your-training-bucket/jobs/my-job/config.yaml
```

### Remove job:
```bash
aws s3 rm s3://your-training-bucket/jobs/my-job/ --recursive
```

---

## Costs

| Resource | Cost |
|----------|------|
| S3 | ~$0.023/GB/month |
| EFS | ~$0.30/GB/month |
| Lambda | ~$0.0000001 per invocation |
| g5.xlarge spot | ~$0.40-0.60/hour |
| SNS | ~$0.0001 per SMS |

Typical job (120 epochs, medium dataset): **$2-5**

---

## Troubleshooting

### Lambda not triggering
- Check S3 event configuration
- Check Lambda permissions
- Check CloudWatch logs

### Spot not starting
- Check spot capacity in your region/AZ
- Try different instance type
- Check IAM role permissions

### Training failing
- Check SNS for error messages
- SSH to instance (if still running) and check logs
- Check EFS for epoch.txt (shows last progress)

### Stuck at epoch
- Watchdog should kill after stall_hours
- Check if GPU is actually being used
- Check for OOM errors
