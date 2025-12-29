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
import atexit
import requests
import boto3
import yaml
from pathlib import Path
from datetime import datetime
from functools import wraps
from botocore.exceptions import ClientError
from botocore.config import Config

# Safety net: track clean exit
_clean_exit = False


def get_required_env(key):
    """Get required environment variable or raise clear error."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# Environment variables (set by user data)
JOB_ID = get_required_env('JOB_ID')
S3_BUCKET = get_required_env('S3_BUCKET')
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', '')


def ntfy(message):
    """Send notification via ntfy.sh (if configured)."""
    if not NTFY_TOPIC:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            timeout=5
        )
    except Exception:
        pass  # Don't fail training on notification errors

# Local paths (everything on instance storage, synced to/from S3)
WORK_DIR = Path('/home/ubuntu/training')
DATASET_DIR = WORK_DIR / 'dataset'
WEIGHTS_DIR = WORK_DIR / 'weights'
CHECKPOINT_FILE = WEIGHTS_DIR / 'train' / 'weights' / 'last.pt'
EPOCH_FILE = WORK_DIR / 'epoch.txt'

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


def _on_exit():
    """Safety net - notify and terminate if we didn't exit cleanly."""
    global _clean_exit
    if not _clean_exit:
        try:
            ntfy(f"💀 [{JOB_ID}] train.py died unexpectedly - check logs")
        except Exception:
            pass
        # Note: instance termination handled by bash trap, but log the event
        print("ERROR: Exiting without clean_exit flag set", file=sys.stderr)


atexit.register(_on_exit)

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
    'save_period': -1,  # Only save best.pt and last.pt
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
    global _clean_exit
    print(f"{JOB_ID} started on {get_instance_type()}")

    try:
        # Check if already complete (previous instance finished but didn't cancel spot)
        if job_already_complete():
            print(f"Job {JOB_ID} already complete. Cleaning up.")
            _clean_exit = True
            cleanup_and_terminate()
            return

        # Check if job still exists in S3
        if not job_exists():
            print(f"Job {JOB_ID} no longer exists in S3. Cleaning up.")
            _clean_exit = True
            cleanup_and_terminate()
            return
        
        # Setup work directory
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        
        # Pull config
        config = pull_config()
        
        # Pull dataset if needed
        pull_dataset_if_needed()

        # Validate and fix dataset paths (generates fresh data.yaml)
        data_yaml = validate_and_fix_dataset()

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
        metrics = train(config, start_epoch, data_yaml)
        
        # Stop watchdog
        watchdog_stop.set()
        
        # Upload results
        upload_weights(config, metrics)

        # Log completion
        recall = metrics.get('recall', 0)
        mAP50 = metrics.get('mAP50', 0)
        training_time = metrics.get('training_time', 0)
        time_str = format_duration(training_time) if training_time else "?"
        print(f"{JOB_ID} complete: {recall:.1%} recall, {mAP50:.1%} mAP50 in {time_str}")
        ntfy(f"✅ [{JOB_ID}] Complete in {time_str} - {recall:.1%} recall, {mAP50:.1%} mAP50")

        # Mark clean exit and terminate
        _clean_exit = True
        cleanup_and_terminate()

    except NaNDetected as e:
        watchdog_stop.set()
        print(f"{JOB_ID} failed: NaN at epoch {e.epoch}")
        ntfy(f"❌ [{JOB_ID}] Training failed: NaN detected at epoch {e.epoch}")
        cleanup_local()
        # Mark clean exit (we handled it) and terminate
        _clean_exit = True
        cleanup_and_terminate()

    except Exception as e:
        watchdog_stop.set()
        error_msg = str(e)
        print(f"{JOB_ID} failed: {error_msg}")
        # Send full error in multiple ntfy messages if needed
        ntfy(f"❌ [{JOB_ID}] Training failed: {error_msg[:200]}")
        if len(error_msg) > 200:
            ntfy(f"...{error_msg[200:400]}")
        # Mark clean exit (we handled it) and terminate
        _clean_exit = True
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
        print("Dataset already present locally")
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


def validate_and_fix_dataset():
    """Validate dataset and generate fresh data.yaml with correct paths."""
    print(f"Dataset directory: {DATASET_DIR}")

    # Check required directories exist
    train_images_dir = DATASET_DIR / 'train' / 'images'
    valid_images_dir = DATASET_DIR / 'valid' / 'images'

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Missing: {train_images_dir}")
    if not valid_images_dir.exists():
        raise FileNotFoundError(f"Missing: {valid_images_dir}")

    # Count images
    train_images = list(train_images_dir.glob('*.[jJ][pP][gG]')) + list(train_images_dir.glob('*.[pP][nN][gG]'))
    valid_images = list(valid_images_dir.glob('*.[jJ][pP][gG]')) + list(valid_images_dir.glob('*.[pP][nN][gG]'))
    print(f"Train images: {len(train_images)}")
    print(f"Valid images: {len(valid_images)}")

    if len(train_images) == 0:
        raise ValueError(f"No images found in {train_images_dir}")

    # Read original data.yaml for class names only
    data_yaml = DATASET_DIR / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml) as f:
            orig = yaml.safe_load(f)
        names = orig.get('names', ['object'])
    else:
        names = ['object']  # Default if no data.yaml

    # Write fresh data.yaml with absolute paths
    fresh_config = {
        'path': str(DATASET_DIR.resolve()),
        'train': str(train_images_dir.resolve()),
        'val': str(valid_images_dir.resolve()),
        'names': names,
        'nc': len(names),
    }
    with open(data_yaml, 'w') as f:
        yaml.dump(fresh_config, f)

    print(f"Data config: {fresh_config}")

    # Notify dataset stats
    class_str = ', '.join(names) if len(names) <= 3 else f"{len(names)} classes"
    ntfy(f"📁 [{JOB_ID}] Dataset: {len(train_images)} train / {len(valid_images)} valid ({class_str})")

    return data_yaml


def get_gpu_info():
    """Get GPU name and memory."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{gpu_name} ({gpu_mem:.0f}GB)"
    except Exception:
        pass
    return "unknown GPU"


def format_duration(seconds):
    """Format seconds as human readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


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


def train(config, start_epoch, data_yaml):
    """Run YOLO training."""
    from ultralytics import YOLO

    # Load model
    model_name = config.get('model', 'yolo12m.pt')
    total_epochs = config.get('epochs', 120)
    patience = config.get('patience', TRAIN_DEFAULTS['patience'])
    gpu_info = get_gpu_info()

    # Track training start time
    training_start_time = time.time()
    epoch_1_time = None  # Will be set after epoch 1

    if start_epoch > 0 and CHECKPOINT_FILE.exists():
        print(f"Resuming from checkpoint at epoch {start_epoch}")
        ntfy(f"🔄 [{JOB_ID}] Resuming from epoch {start_epoch}/{total_epochs} on {gpu_info}")
        model = YOLO(str(CHECKPOINT_FILE))
    else:
        print(f"Starting fresh with {model_name}")
        ntfy(f"🏋️ [{JOB_ID}] Training starting: {model_name}, {total_epochs} epochs on {gpu_info}")
        model = YOLO(model_name)

    # Epoch tracking callback
    def on_train_epoch_end(trainer):
        nonlocal epoch_1_time
        epoch = trainer.epoch
        write_epoch(epoch)

        # Check for NaN
        loss = trainer.loss
        if loss is not None and (math.isnan(loss) or loss > 100):
            raise NaNDetected(epoch)

        # Send progress notification: epoch 1 (with ETA), then every 10
        if epoch == 1:
            epoch_1_time = time.time() - training_start_time
            remaining_epochs = total_epochs - epoch
            eta_seconds = epoch_1_time * remaining_epochs
            loss_str = f"{float(loss):.3f}" if loss is not None else "?"
            ntfy(f"📊 [{JOB_ID}] Epoch 1/{total_epochs} - loss: {loss_str}, ETA: ~{format_duration(eta_seconds)}")
        elif epoch > 0 and epoch % 10 == 0:
            loss_str = f"{float(loss):.3f}" if loss is not None else "?"
            ntfy(f"📊 [{JOB_ID}] Epoch {epoch}/{total_epochs} - loss: {loss_str}")

        # Check for spot interruption - stop training gracefully
        if spot_interrupted.is_set():
            print(f"Stopping training at epoch {epoch} due to spot interruption")
            ntfy(f"⚠️ [{JOB_ID}] Spot interruption at epoch {epoch}, saving checkpoint...")
            trainer.stop = True

    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    # Build training args: defaults merged with user config
    train_args = {**TRAIN_DEFAULTS, **config}

    # Override/add computed values
    train_args.update({
        'data': str(data_yaml),
        'project': str(WEIGHTS_DIR),
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

    # Calculate training duration
    training_duration = time.time() - training_start_time
    duration_str = format_duration(training_duration)

    # Mark complete
    write_epoch('complete')

    # Extract metrics
    metrics = {}
    final_epoch = total_epochs
    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        metrics['recall'] = rd.get('metrics/recall(B)', 0)
        metrics['precision'] = rd.get('metrics/precision(B)', 0)
        metrics['mAP50'] = rd.get('metrics/mAP50(B)', 0)
        metrics['mAP50-95'] = rd.get('metrics/mAP50-95(B)', 0)

    # Check for early stopping (training stopped before total_epochs)
    if hasattr(results, 'epoch'):
        final_epoch = results.epoch + 1  # epoch is 0-indexed
        if final_epoch < total_epochs:
            ntfy(f"⏹️ [{JOB_ID}] Early stopped at epoch {final_epoch}/{total_epochs} (patience={patience})")

    # Add training stats to metrics for logging
    metrics['training_time'] = training_duration
    metrics['final_epoch'] = final_epoch

    return metrics


@retry_on_transient()
def upload_weights(config, metrics):
    """Upload trained weights and results to S3."""
    # best.pt is always in YOLO's output location
    best_pt = WEIGHTS_DIR / 'train' / 'weights' / 'best.pt'

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
    checkpoint = WEIGHTS_DIR / 'train' / 'weights' / 'last.pt'

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
                    ntfy(f"⚠️ [{JOB_ID}] Training stalled - no progress in {age_hours:.1f}h, terminating")
                    cleanup_local()
                    cleanup_and_terminate()
        except Exception as e:
            print(f"Watchdog error: {e}")


def cleanup_local():
    """Remove work directory (instance terminates anyway, but be tidy)."""
    try:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
            print(f"Cleaned up: {WORK_DIR}")
    except Exception as e:
        print(f"Failed to cleanup: {e}")


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
            # Only return on success - 404 means metadata not available
            if response.status_code == 200:
                return response.text
            return None
        # Fall back to IMDSv1
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return response.text
        return None
    except requests.RequestException:
        return None


def get_instance_type():
    """Get current instance type."""
    return get_metadata('instance-type') or "unknown"


if __name__ == '__main__':
    main()
