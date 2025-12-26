#!/bin/bash
cd "$(dirname "$0")"

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
