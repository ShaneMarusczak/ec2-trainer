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
from functools import wraps
from botocore.exceptions import ClientError
from botocore.config import Config


def get_required_env(key):
    """Get required environment variable or raise clear error."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# Environment variables (set by user data)
JOB_ID = get_required_env('JOB_ID')
S3_BUCKET = get_required_env('S3_BUCKET')
EFS_ID = get_required_env('EFS_ID')

# Paths
EFS_ROOT = Path('/mnt/efs')
JOB_DIR = EFS_ROOT / JOB_ID
EPOCH_FILE = JOB_DIR / 'epoch.txt'
DATASET_DIR = JOB_DIR / 'dataset'
LOCAL_WEIGHTS_DIR = JOB_DIR / 'weights'
# YOLO saves checkpoints here
CHECKPOINT_FILE = LOCAL_WEIGHTS_DIR / 'train' / 'weights' / 'last.pt'

# AWS clients with retry config
s3_config = Config(
    connect_timeout=30,
    read_timeout=300,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)
s3 = boto3.client('s3', config=s3_config)
ec2 = boto3.client('ec2')

# Watchdog state
watchdog_stop = threading.Event()
spot_interrupted = threading.Event()

# Training defaults (overridden by config.yaml)
TRAIN_DEFAULTS = {
    'epochs': 120,
    'batch': 16,
    'imgsz': 640,
    'patience': 20,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'warmup_epochs': 5,
    'dropout': 0.1,
    'box': 0.5,
    'cls': 3.0,
    'dfl': 0.5,
    'conf': 0.01,
    'iou': 0.45,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 15,
    'translate': 0.1,
    'scale': 0.5,
    'mosaic': 1.0,
    'mixup': 0.15,
    'workers': 8,
    'close_mosaic': 10,
    'copy_paste': 0.3,
    'save_period': 10,
}


def retry_on_transient(max_attempts=3, backoff=2):
    """Retry decorator for transient S3 errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    code = e.response['Error']['Code']
                    if code in ('Throttling', 'RequestTimeout', 'ServiceUnavailable', 'SlowDown'):
                        last_exception = e
                        time.sleep(backoff ** attempt)
                        continue
                    raise
            raise last_exception
        return wrapper
    return decorator


def s3_key_exists(key):
    """Check if a key exists in S3."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def main():
    """Main entry point."""
    print(f"{JOB_ID} started on {get_instance_type()}")

    try:
        # Check if already complete (previous instance finished but didn't cancel spot)
        if job_already_complete():
            print(f"Job {JOB_ID} already complete. Cleaning up.")
            cleanup_and_terminate()
            return

        # Check if job still exists in S3
        if not job_exists():
            print(f"Job {JOB_ID} no longer exists in S3. Cleaning up.")
            cleanup_and_terminate()
            return
        
        # Setup job directory
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Pull config
        config = pull_config()
        
        # Pull dataset if needed
        pull_dataset_if_needed()

        # Validate dataset structure
        validate_dataset()

        # Get starting epoch
        start_epoch = read_epoch()
        print(f"Starting from epoch {start_epoch}")
        
        # Start watchdog and spot interruption monitor
        stall_hours = config.get('stall_hours', 2)
        watchdog_thread = threading.Thread(
            target=watchdog,
            args=(stall_hours,),
            daemon=True
        )
        watchdog_thread.start()

        spot_monitor = threading.Thread(
            target=spot_interruption_monitor,
            daemon=True
        )
        spot_monitor.start()
        
        # Train
        metrics = train(config, start_epoch)
        
        # Stop watchdog
        watchdog_stop.set()
        
        # Upload results
        upload_weights(config, metrics)

        # Log completion
        recall = metrics.get('recall', 0)
        mAP50 = metrics.get('mAP50', 0)
        print(f"{JOB_ID} complete: {recall:.1%} recall, {mAP50:.1%} mAP50")

        # Cleanup and terminate
        cleanup_and_terminate()
        
    except NaNDetected as e:
        watchdog_stop.set()
        print(f"{JOB_ID} failed: NaN at epoch {e.epoch}")
        cleanup_efs()
        cleanup_and_terminate()

    except Exception as e:
        watchdog_stop.set()
        print(f"{JOB_ID} failed: {str(e)[:100]}")
        cleanup_and_terminate()


class NaNDetected(Exception):
    """Raised when NaN is detected in training."""
    def __init__(self, epoch):
        self.epoch = epoch


def job_already_complete():
    """Check if weights already exist in S3."""
    return s3_key_exists(f"weights/{JOB_ID}/best.pt")


def job_exists():
    """Check if job still exists in S3."""
    return s3_key_exists(f"jobs/{JOB_ID}/config.yaml")


@retry_on_transient()
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

    # Use aws s3 sync for parallel downloads
    import subprocess
    cmd = [
        'aws', 's3', 'sync',
        f's3://{S3_BUCKET}/jobs/{JOB_ID}/dataset/',
        str(DATASET_DIR),
        '--only-show-errors'
    ]
    subprocess.run(cmd, check=True)

    print(f"Dataset downloaded to {DATASET_DIR}")


def validate_dataset():
    """Validate dataset has required structure."""
    data_yaml = DATASET_DIR / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Dataset must have data.yaml in root."
        )

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    required = ['names', 'train']
    missing = [k for k in required if k not in data_config]
    if missing:
        raise ValueError(f"data.yaml missing required keys: {missing}")

    print(f"Dataset validated: {len(data_config['names'])} classes")


def read_epoch():
    """Read current epoch from checkpoint file."""
    if EPOCH_FILE.exists():
        content = EPOCH_FILE.read_text().strip()
        if content == 'complete':
            return -1  # Signal that it's already done
        try:
            return int(content)
        except ValueError:
            return 0  # Corrupted file, start over
    return 0


def write_epoch(epoch):
    """Atomic write to epoch checkpoint file."""
    tmp_file = EPOCH_FILE.with_suffix('.tmp')
    tmp_file.write_text(str(epoch))
    tmp_file.replace(EPOCH_FILE)  # Atomic on POSIX


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

    # data.yaml must be in standard location (validated earlier)
    data_yaml = DATASET_DIR / 'data.yaml'

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

        # Check for spot interruption - stop training gracefully
        if spot_interrupted.is_set():
            print(f"Stopping training at epoch {epoch} due to spot interruption")
            trainer.stop = True

    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    # Build training args: defaults merged with user config
    train_args = {**TRAIN_DEFAULTS, **config}

    # Override/add computed values
    train_args.update({
        'data': str(data_yaml),
        'project': str(LOCAL_WEIGHTS_DIR),
        'name': 'train',
        'exist_ok': True,
    })

    # Resume if needed
    if start_epoch > 0:
        train_args['resume'] = True

    # Remove non-YOLO keys from config
    non_yolo_keys = ['model', 'instance_type', 'stall_hours']
    for key in non_yolo_keys:
        train_args.pop(key, None)

    # Train
    results = model.train(**train_args)

    # Mark complete
    write_epoch('complete')

    # Extract metrics
    metrics = {}
    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics['recall'] = rd.get('metrics/recall(B)', 0)
        metrics['precision'] = rd.get('metrics/precision(B)', 0)
        metrics['mAP50'] = rd.get('metrics/mAP50(B)', 0)
        metrics['mAP50-95'] = rd.get('metrics/mAP50-95(B)', 0)

    return metrics


@retry_on_transient()
def upload_weights(config, metrics):
    """Upload trained weights and results to S3."""
    # best.pt is always in YOLO's output location
    best_pt = LOCAL_WEIGHTS_DIR / 'train' / 'weights' / 'best.pt'

    if best_pt.exists():
        s3.upload_file(str(best_pt), S3_BUCKET, f"weights/{JOB_ID}/best.pt")
        print("Uploaded best.pt")
    else:
        print("WARNING: best.pt not found")

    # Upload config directly (no temp file needed)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"weights/{JOB_ID}/config.yaml",
        Body=yaml.dump(config)
    )

    # Upload results directly
    results = {
        'job_id': JOB_ID,
        'completed_at': datetime.now().isoformat(),
        'metrics': metrics,
        'config': config
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"weights/{JOB_ID}/results.yaml",
        Body=yaml.dump(results)
    )

    print(f"Uploaded results to s3://{S3_BUCKET}/weights/{JOB_ID}/")


def spot_interruption_monitor():
    """Monitor for spot instance interruption notice (2-min warning)."""
    while not watchdog_stop.is_set():
        time.sleep(5)  # Check every 5 seconds

        if watchdog_stop.is_set():
            break

        try:
            # Check for spot interruption notice
            action = get_metadata('spot/instance-action')
            if action:
                print(f"SPOT INTERRUPTION: {action}")
                spot_interrupted.set()
                # Don't terminate - let training save checkpoint and exit gracefully
                break
        except Exception:
            pass  # No interruption notice


def watchdog(stall_hours):
    """Monitor training progress via checkpoint file mtime."""
    checkpoint = LOCAL_WEIGHTS_DIR / 'train' / 'weights' / 'last.pt'

    while not watchdog_stop.is_set():
        time.sleep(600)  # Check every 10 minutes

        if watchdog_stop.is_set():
            break

        try:
            # Check if training marked complete
            if EPOCH_FILE.exists() and EPOCH_FILE.read_text().strip() == 'complete':
                break

            # Check checkpoint staleness
            if checkpoint.exists():
                age_hours = (time.time() - checkpoint.stat().st_mtime) / 3600
                if age_hours > stall_hours:
                    print(f"{JOB_ID} stalled - no checkpoint update in {age_hours:.1f}h")
                    cleanup_efs()
                    cleanup_and_terminate()
        except Exception as e:
            print(f"Watchdog error: {e}")


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
        instance_id = get_metadata('instance-id')
        if not instance_id:
            print("Could not get instance ID")
            sys.exit(1)

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


def get_imds_token():
    """Get IMDSv2 token for metadata access."""
    try:
        response = requests.put(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '300'},
            timeout=2
        )
        return response.text
    except requests.RequestException:
        return None


def get_metadata(path):
    """Get instance metadata (supports both IMDSv1 and IMDSv2)."""
    url = f'http://169.254.169.254/latest/meta-data/{path}'
    try:
        # Try IMDSv2 first
        token = get_imds_token()
        if token:
            response = requests.get(
                url,
                headers={'X-aws-ec2-metadata-token': token},
                timeout=2
            )
            return response.text
        # Fall back to IMDSv1
        return requests.get(url, timeout=2).text
    except requests.RequestException:
        return None


def get_instance_type():
    """Get current instance type."""
    return get_metadata('instance-type') or "unknown"


if __name__ == '__main__':
    main()
