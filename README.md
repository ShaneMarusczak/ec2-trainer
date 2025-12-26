# EC2 YOLO Training

Train YOLO models on AWS spot instances. Upload a job, get trained weights.

## Usage

```bash
./run.sh    # menu: start job or check status
```

## Flow

```
prep.py → S3 + spot request → EC2 trains → S3 weights → pull.py
```

## First Run

First `./run.sh` auto-runs setup.py which creates all AWS infrastructure:
- S3 bucket
- EFS file system + mount target
- Security group (locked down - EFS only)
- IAM role + instance profile

Config saved to `~/.ec2-trainer.yaml`

## Files

```
run.sh      - Entry point (auto-setup on first run)
setup.py    - One-time AWS infrastructure setup
prep.py     - Upload datasets, start training
pull.py     - Check status, sync weights
train.py    - Runs on EC2 (bundled with each job)
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
