# EC2 YOLO Training Pipeline

Automatic YOLO training on AWS spot instances. Upload a job, get trained weights.

## How It Works

```
You upload job to S3
  → S3 triggers Lambda
  → Lambda creates spot request (1:1 per job)
  → Spot instance trains your model
  → If interrupted: respawns and resumes
  → When complete: uploads weights, terminates
  → You get a text notification
```

## Quick Start

1. **Setup AWS** (one time): Follow `docs/setup.md`

2. **Create a job folder:**
```
my-detector/
  config.yaml      # Training parameters
  dataset/
    train/
      images/
      labels/
    valid/
      images/
      labels/
    data.yaml
```

3. **Upload:**
```bash
aws s3 cp my-detector s3://your-bucket/jobs/my-detector/ --recursive
```

4. **Wait for text:** "my-detector complete: 84.3% recall"

5. **Download weights:**
```bash
aws s3 cp s3://your-bucket/weights/my-detector/best.pt ./
```

## Files

```
lambda/
  handler.py          # S3 trigger → create spot request

trainer/
  train.py            # Main training script
  bootstrap.sh        # EC2 startup script
  requirements.txt

config/
  example.yaml        # Example configuration

docs/
  setup.md            # AWS setup instructions
```

## Features

- **1:1 job binding**: Each spot request is bound to one job
- **Auto-resume**: Spot interruption → respawns → continues from checkpoint
- **NaN detection**: Training failure → cleanup → notification
- **Watchdog**: Stuck training → auto-kill after N hours
- **Zero idle cost**: Nothing runs when queue is empty

## Cost

- g5.xlarge spot: ~$0.40-0.60/hour
- Typical job: $2-5
- Idle: $0

## Configuration

See `config/example.yaml` for all options.

Key settings:
```yaml
model: yolo12m.pt       # Which YOLO model
instance_type: g5.xlarge  # EC2 size
epochs: 120             # Training length
cls: 3.0                # Classification weight
stall_hours: 2          # Watchdog timeout
```

## Commands

```bash
# Add job
aws s3 cp ./my-job s3://bucket/jobs/my-job/ --recursive

# Check progress
aws s3 ls s3://bucket/weights/

# Get weights
aws s3 cp s3://bucket/weights/my-job/best.pt ./

# Force retrain
aws s3 rm s3://bucket/weights/my-job/ --recursive
aws s3 cp config.yaml s3://bucket/jobs/my-job/config.yaml

# Cancel job (remove entirely)
aws s3 rm s3://bucket/jobs/my-job/ --recursive
```

## Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────────┐
│   S3    │────>│ Lambda  │────>│ Spot Request│
│  jobs/  │     │         │     │   (1:1)     │
└─────────┘     └─────────┘     └──────┬──────┘
                                       │
                                       v
┌─────────┐     ┌─────────┐     ┌─────────────┐
│   S3    │<────│   EFS   │<────│     EC2     │
│ weights/│     │checkpoint│    │   trains    │
└─────────┘     └─────────┘     └─────────────┘
                                       │
                                       v
                                ┌─────────────┐
                                │    SNS      │
                                │  (notify)   │
                                └─────────────┘
```
