#!/bin/bash
# Setup script to organize data for ViViT training

# Configuration - adjust these paths as needed
VIDEO_SOURCE_DIR="/home/retrocausal-train/Desktop/TR/VideoTransformer-pytorch/ssv2/20bn-something-something-v2"
TRAIN_DATA_DIR="./data/train"
VAL_DATA_DIR="./data/val"

echo "Setting up data directories..."

# Create directories
mkdir -p "$TRAIN_DATA_DIR/videos"
mkdir -p "$VAL_DATA_DIR/videos"

# Copy CSV files to the right locations
echo "Copying label CSV files..."
cp train_label.csv "$TRAIN_DATA_DIR/label.csv"
cp val_label.csv "$VAL_DATA_DIR/label.csv"

# Extract video filenames from CSV and create symlinks (or copy if you prefer)
echo "Creating symlinks for training videos..."
while IFS=',' read -r fname label || [ -n "$fname" ]; do
    # Skip header line
    if [ "$fname" = "fname" ]; then
        continue
    fi
    # Remove any whitespace
    fname=$(echo "$fname" | tr -d ' ')
    if [ -f "$VIDEO_SOURCE_DIR/$fname" ]; then
        ln -sf "$VIDEO_SOURCE_DIR/$fname" "$TRAIN_DATA_DIR/videos/$fname"
    else
        echo "Warning: Video not found: $VIDEO_SOURCE_DIR/$fname"
    fi
done < train_label.csv

echo "Creating symlinks for validation videos..."
while IFS=',' read -r fname label || [ -n "$fname" ]; do
    # Skip header line
    if [ "$fname" = "fname" ]; then
        continue
    fi
    # Remove any whitespace
    fname=$(echo "$fname" | tr -d ' ')
    if [ -f "$VIDEO_SOURCE_DIR/$fname" ]; then
        ln -sf "$VIDEO_SOURCE_DIR/$fname" "$VAL_DATA_DIR/videos/$fname"
    else
        echo "Warning: Video not found: $VIDEO_SOURCE_DIR/$fname"
    fi
done < val_label.csv

echo ""
echo "✓ Data setup complete!"
echo ""
echo "Directory structure:"
echo "  $TRAIN_DATA_DIR/videos/  - Training videos"
echo "  $TRAIN_DATA_DIR/label.csv - Training labels"
echo "  $VAL_DATA_DIR/videos/    - Validation videos"
echo "  $VAL_DATA_DIR/label.csv  - Validation labels"
echo ""
echo "To train, run:"
echo "  python train_vivit.py --dataset custom --data_dir $TRAIN_DATA_DIR --test_dir $VAL_DATA_DIR --name ssv2_train --pretrained_dir /path/to/ViT-B_16.npz --num_classes 174 --num_frames 32"


