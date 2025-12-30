#!/usr/bin/env python3
"""
Spaghetti Detector Service for Jetson Orin

Runs YOLO inference on camera feed, pauses OctoPrint on detection.
Triggered by OctoPrint webhooks on print start/stop.

Usage:
    uvicorn detector:app --host 0.0.0.0 --port 8000
"""

import threading
import time
from pathlib import Path

# Auto-install dependencies
import subprocess
import sys
for pkg in ['fastapi', 'uvicorn', 'httpx', 'opencv-python']:
    try:
        __import__(pkg.replace('-', '_').split('[')[0])
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import cv2
import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# =============================================================================
# Configuration - edit these
# =============================================================================
MODEL_PATH = Path("best.pt")  # Your trained YOLO weights
CAMERA = "/dev/video0"  # or "http://octopi.local/webcam/?action=stream"
OCTOPRINT_URL = "http://octopi.local"
OCTOPRINT_API_KEY = "YOUR_API_KEY"  # Get from OctoPrint settings
CONFIDENCE_THRESHOLD = 0.5
CHECK_INTERVAL = 1.0  # seconds between inference

# =============================================================================
# App
# =============================================================================
app = FastAPI(title="Spaghetti Detector")

# State
monitoring = False
monitor_thread = None
model = None
detections = []


class WebhookEvent(BaseModel):
    event: str
    payload: dict = {}


@app.on_event("startup")
def load_model():
    """Load YOLO model on startup."""
    global model
    if MODEL_PATH.exists():
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")


@app.post("/start")
def start():
    """Start monitoring camera for spaghetti."""
    global monitoring, monitor_thread

    if monitoring:
        return {"status": "already running"}

    if model is None:
        return {"status": "error", "message": "model not loaded"}

    monitoring = True
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    return {"status": "started"}


@app.post("/stop")
def stop():
    """Stop monitoring."""
    global monitoring
    monitoring = False
    return {"status": "stopped"}


@app.get("/status")
def status():
    """Get current status."""
    return {
        "monitoring": monitoring,
        "model_loaded": model is not None,
        "recent_detections": detections[-10:]  # last 10
    }


@app.post("/webhook")
def octoprint_webhook(event: WebhookEvent):
    """Handle OctoPrint webhook events."""
    e = event.event

    if e == "PrintStarted":
        return start()
    elif e in ("PrintDone", "PrintFailed", "PrintCancelled"):
        return stop()

    return {"status": "ignored", "event": e}


def monitor_loop():
    """Main monitoring loop - runs inference on camera frames."""
    global monitoring, detections

    print(f"Starting monitor, camera: {CAMERA}")
    cap = cv2.VideoCapture(CAMERA)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera: {CAMERA}")
        monitoring = False
        return

    consecutive_detections = 0
    TRIGGER_THRESHOLD = 3  # require N consecutive detections to avoid false positives

    while monitoring:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            time.sleep(1)
            continue

        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        if results[0].boxes:
            consecutive_detections += 1
            detection = {
                "time": time.time(),
                "count": len(results[0].boxes),
                "confidence": float(results[0].boxes.conf.max()),
            }
            detections.append(detection)
            print(f"Detection {consecutive_detections}/{TRIGGER_THRESHOLD}: {detection}")

            if consecutive_detections >= TRIGGER_THRESHOLD:
                print("SPAGHETTI DETECTED - pausing print!")
                pause_print()
                monitoring = False
                break
        else:
            consecutive_detections = 0

        time.sleep(CHECK_INTERVAL)

    cap.release()
    print("Monitor stopped")


def pause_print():
    """Pause the print via OctoPrint API."""
    try:
        response = httpx.post(
            f"{OCTOPRINT_URL}/api/job",
            json={"command": "pause", "action": "pause"},
            headers={"X-Api-Key": OCTOPRINT_API_KEY},
            timeout=10
        )
        print(f"Pause response: {response.status_code}")
    except Exception as e:
        print(f"Failed to pause print: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
