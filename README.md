# EC2 YOLO Training

Train YOLO models on AWS spot instances. Upload a job, get trained weights.

## Usage

```bash
python prep.py    # datasets → config → upload → start training
# [wait for SMS]
python pull.py    # sync weights to local
```

## Flow

```
prep.py → S3 + spot request → EC2 trains → S3 weights → pull.py
```

## First Run

prep.py will ask for AWS infrastructure config (saved to `~/.ec2-trainer.yaml`):
- EFS ID
- SNS Topic ARN
- Subnet ID
- Security Group ID
- IAM Instance Profile name
- AMI ID

## Files

```
prep.py     - Upload datasets, start training
train.py    - Runs on EC2 (bundled with each job)
pull.py     - Sync weights from S3
```

## Features

- **Self-contained jobs**: Each upload includes train.py
- **Spot interruption recovery**: Resumes from EFS checkpoint
- **Watchdog**: Kills stalled training
- **NaN detection**: Stops on bad loss values

## Cost

- g5.xlarge spot: ~$0.40-0.60/hour
- Typical job: $2-5
- Idle: $0
