"""
EC2 Training Script

Single job trainer with:
- Resume from checkpoint
- Watchdog for stalled training
- NaN detection
- Self-cleanup on complete/fail
"""

import os
import sys
import time
import math
import shutil
import threading
import requests
import boto3
import yaml
from pathlib import Path
from datetime import datetime

# Environment variables (set by user data)
JOB_ID = os.environ['JOB_ID']
S3_BUCKET = os.environ['S3_BUCKET']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
EFS_ID = os.environ['EFS_ID']

# Paths
EFS_ROOT = Path('/mnt/efs')
JOB_DIR = EFS_ROOT / JOB_ID
EPOCH_FILE = JOB_DIR / 'epoch.txt'
CHECKPOINT_FILE = JOB_DIR / 'last.pt'
DATASET_DIR = JOB_DIR / 'dataset'
LOCAL_WEIGHTS_DIR = JOB_DIR / 'weights'

# AWS clients
s3 = boto3.client('s3')
ec2 = boto3.client('ec2')
sns = boto3.client('sns')

# Watchdog state
watchdog_stop = threading.Event()


def main():
    """Main entry point."""
    notify(f"{JOB_ID} started on {get_instance_type()}")
    
    try:
        # Check if already complete (previous instance finished but didn't cancel spot)
        if job_already_complete():
            print(f"Job {JOB_ID} already complete. Cleaning up.")
            notify(f"{JOB_ID} already complete - cleaning up stale spot")
            cleanup_and_terminate()
            return
        
        # Check if job still exists in S3
        if not job_exists():
            print(f"Job {JOB_ID} no longer exists in S3. Cleaning up.")
            notify(f"{JOB_ID} deleted - cleaning up")
            cleanup_and_terminate()
            return
        
        # Setup job directory
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Pull config
        config = pull_config()
        
        # Pull dataset if needed
        pull_dataset_if_needed()
        
        # Get starting epoch
        start_epoch = read_epoch()
        print(f"Starting from epoch {start_epoch}")
        
        # Start watchdog
        stall_hours = config.get('stall_hours', 2)
        watchdog_thread = threading.Thread(
            target=watchdog,
            args=(stall_hours,),
            daemon=True
        )
        watchdog_thread.start()
        
        # Train
        metrics = train(config, start_epoch)
        
        # Stop watchdog
        watchdog_stop.set()
        
        # Upload results
        upload_weights(config, metrics)
        
        # Notify success
        recall = metrics.get('recall', 0)
        mAP50 = metrics.get('mAP50', 0)
        notify(f"{JOB_ID} complete: {recall:.1%} recall, {mAP50:.1%} mAP50")
        
        # Cleanup and terminate
        cleanup_and_terminate()
        
    except NaNDetected as e:
        watchdog_stop.set()
        notify(f"{JOB_ID} failed: NaN at epoch {e.epoch}")
        cleanup_efs()
        cleanup_and_terminate()
        
    except Exception as e:
        watchdog_stop.set()
        notify(f"{JOB_ID} failed: {str(e)[:100]}")
        cleanup_and_terminate()


class NaNDetected(Exception):
    """Raised when NaN is detected in training."""
    def __init__(self, epoch):
        self.epoch = epoch


def job_already_complete():
    """Check if weights already exist in S3."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=f"weights/{JOB_ID}/best.pt")
        return True
    except:
        return False


def job_exists():
    """Check if job still exists in S3."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=f"jobs/{JOB_ID}/config.yaml")
        return True
    except:
        return False


def pull_config():
    """Download config.yaml from S3."""
    response = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{JOB_ID}/config.yaml")
    config = yaml.safe_load(response['Body'].read())
    print(f"Config loaded: {config}")
    return config


def pull_dataset_if_needed():
    """Download dataset from S3 if not already present."""
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print("Dataset already present in EFS")
        return
    
    print("Pulling dataset from S3...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # List and download all dataset files
    paginator = s3.get_paginator('list_objects_v2')
    prefix = f"jobs/{JOB_ID}/dataset/"
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            relative_path = key[len(prefix):]
            local_path = DATASET_DIR / relative_path
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(local_path))
    
    print(f"Dataset downloaded to {DATASET_DIR}")


def read_epoch():
    """Read current epoch from checkpoint file."""
    if EPOCH_FILE.exists():
        content = EPOCH_FILE.read_text().strip()
        if content == 'complete':
            return -1  # Signal that it's already done
        return int(content)
    return 0


def write_epoch(epoch):
    """Write current epoch to checkpoint file."""
    EPOCH_FILE.write_text(str(epoch))


def train(config, start_epoch):
    """Run YOLO training."""
    from ultralytics import YOLO
    
    # Load model
    model_name = config.get('model', 'yolo12m.pt')
    
    if start_epoch > 0 and CHECKPOINT_FILE.exists():
        print(f"Resuming from checkpoint at epoch {start_epoch}")
        model = YOLO(str(CHECKPOINT_FILE))
    else:
        print(f"Starting fresh with {model_name}")
        model = YOLO(model_name)
    
    # Find data.yaml
    data_yaml = DATASET_DIR / 'data.yaml'
    if not data_yaml.exists():
        # Try to find it
        for f in DATASET_DIR.rglob('data.yaml'):
            data_yaml = f
            break
    
    # Update data.yaml path to be absolute
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    data_config['path'] = str(data_yaml.parent)
    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    # Epoch tracking callback
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch
        write_epoch(epoch)
        
        # Check for NaN
        loss = trainer.loss
        if loss is not None and (math.isnan(loss) or loss > 100):
            raise NaNDetected(epoch)
    
    # Add callback
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    
    # Training parameters from config
    epochs = config.get('epochs', 120)
    
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'patience': config.get('patience', 20),
        'project': str(LOCAL_WEIGHTS_DIR),
        'name': 'train',
        'exist_ok': True,
        
        # Optimizer
        'optimizer': config.get('optimizer', 'AdamW'),
        'lr0': config.get('lr0', 0.001),
        'warmup_epochs': config.get('warmup_epochs', 5),
        
        # Regularization
        'dropout': config.get('dropout', 0.1),
        
        # Loss weights
        'box': config.get('box', 0.5),
        'cls': config.get('cls', 3.0),
        'dfl': config.get('dfl', 0.5),
        
        # Validation
        'conf': config.get('conf', 0.01),
        'iou': config.get('iou', 0.45),
        
        # Augmentation
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'degrees': config.get('degrees', 15),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'mosaic': config.get('mosaic', 1.0),
        'mixup': config.get('mixup', 0.15),
        
        # Other
        'workers': config.get('workers', 8),
        'close_mosaic': config.get('close_mosaic', 10),
        'copy_paste': config.get('copy_paste', 0.3),
        'save_period': config.get('save_period', 10),
    }
    
    # Resume if needed
    if start_epoch > 0:
        train_args['resume'] = True
    
    # Train
    results = model.train(**train_args)
    
    # Mark complete
    write_epoch('complete')
    
    # Copy best.pt to checkpoint location for upload
    best_pt = LOCAL_WEIGHTS_DIR / 'train' / 'weights' / 'best.pt'
    if best_pt.exists():
        shutil.copy(best_pt, JOB_DIR / 'best.pt')
    
    # Extract metrics
    metrics = {}
    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics['recall'] = rd.get('metrics/recall(B)', 0)
        metrics['precision'] = rd.get('metrics/precision(B)', 0)
        metrics['mAP50'] = rd.get('metrics/mAP50(B)', 0)
        metrics['mAP50-95'] = rd.get('metrics/mAP50-95(B)', 0)
    
    return metrics


def upload_weights(config, metrics):
    """Upload trained weights and results to S3."""
    
    best_pt = JOB_DIR / 'best.pt'
    if not best_pt.exists():
        # Try alternate location
        best_pt = LOCAL_WEIGHTS_DIR / 'train' / 'weights' / 'best.pt'
    
    if best_pt.exists():
        s3.upload_file(str(best_pt), S3_BUCKET, f"weights/{JOB_ID}/best.pt")
        print(f"Uploaded best.pt")
    
    # Upload config for reference
    config_copy = JOB_DIR / 'config.yaml'
    with open(config_copy, 'w') as f:
        yaml.dump(config, f)
    s3.upload_file(str(config_copy), S3_BUCKET, f"weights/{JOB_ID}/config.yaml")
    
    # Upload results
    results = {
        'job_id': JOB_ID,
        'completed_at': datetime.now().isoformat(),
        'metrics': metrics,
        'config': config
    }
    results_file = JOB_DIR / 'results.yaml'
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    s3.upload_file(str(results_file), S3_BUCKET, f"weights/{JOB_ID}/results.yaml")
    
    print(f"Uploaded results to s3://{S3_BUCKET}/weights/{JOB_ID}/")


def watchdog(stall_hours):
    """Monitor training progress, kill if stuck."""
    last_epoch = -1
    last_change = time.time()
    
    while not watchdog_stop.is_set():
        time.sleep(600)  # Check every 10 minutes
        
        if watchdog_stop.is_set():
            break
        
        try:
            current_epoch = read_epoch()
            if current_epoch == -1:  # Complete
                break
                
            if current_epoch != last_epoch:
                last_epoch = current_epoch
                last_change = time.time()
            else:
                stuck_hours = (time.time() - last_change) / 3600
                if stuck_hours > stall_hours:
                    notify(f"{JOB_ID} stuck at epoch {current_epoch} for {stuck_hours:.1f}h - terminating")
                    cleanup_efs()
                    cleanup_and_terminate()
        except Exception as e:
            print(f"Watchdog error: {e}")


def notify(message):
    """Send SNS notification."""
    try:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message,
            Subject=f"Training: {JOB_ID}"
        )
        print(f"Notification sent: {message}")
    except Exception as e:
        print(f"Failed to send notification: {e}")


def cleanup_efs():
    """Remove job directory from EFS."""
    try:
        if JOB_DIR.exists():
            shutil.rmtree(JOB_DIR)
            print(f"Cleaned up EFS: {JOB_DIR}")
    except Exception as e:
        print(f"Failed to cleanup EFS: {e}")


def cleanup_and_terminate():
    """Cancel spot request and terminate instance."""
    try:
        # Get instance ID
        instance_id = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-id',
            timeout=2
        ).text
        
        # Get spot request ID
        response = ec2.describe_spot_instance_requests(
            Filters=[
                {'Name': 'instance-id', 'Values': [instance_id]}
            ]
        )
        
        if response['SpotInstanceRequests']:
            spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            
            # Cancel spot request (prevents respawn)
            ec2.cancel_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            print(f"Cancelled spot request: {spot_request_id}")
        
        # Terminate instance
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"Terminated instance: {instance_id}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)


def get_instance_type():
    """Get current instance type."""
    try:
        return requests.get(
            'http://169.254.169.254/latest/meta-data/instance-type',
            timeout=2
        ).text
    except:
        return "unknown"


if __name__ == '__main__':
    main()
