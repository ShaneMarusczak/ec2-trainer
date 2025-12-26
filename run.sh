#!/bin/bash
cd "$(dirname "$0")"

# Check if setup needed
if [ ! -f ~/.ec2-trainer.yaml ]; then
    echo "============================================================"
    echo "  EC2 YOLO Training - First Time Setup"
    echo "============================================================"
    echo
    python3 setup.py
    exit 0
fi

echo "============================================================"
echo "  EC2 YOLO Training"
echo "============================================================"
echo
echo "  1. Start new job"
echo "  2. Check status / pull weights"
echo
read -p "> " choice

case $choice in
    1) python3 prep.py ;;
    2) python3 pull.py ;;
    *) echo "Cancelled." ;;
esac
