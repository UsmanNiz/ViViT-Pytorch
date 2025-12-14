#!/usr/bin/env python3
"""
Convert Epic Kitchens CSV annotations to simplified CSV format for ViViT training.

Input CSV format (EPIC_100_train.csv / EPIC_100_validation.csv):
    narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,
    start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes

Output CSV format:
    fname,noun_id,verb_id
    P01_11_0.mp4,2,0
    P01_11_1.mp4,2,1
    ...

Usage:
    python convert_epic_kitchens_csv.py \
        --train_csv EPIC_100_train.csv \
        --val_csv EPIC_100_validation.csv \
        --output_dir ./processed_videos
"""

import csv
import argparse
import os


def convert_csv(input_csv_path, output_csv_path):
    """
    Convert Epic Kitchens CSV to simplified format.
    
    Args:
        input_csv_path: Path to input CSV file (EPIC_100_train.csv or EPIC_100_validation.csv)
        output_csv_path: Path to output CSV file
    """
    rows = []
    
    with open(input_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for entry in reader:
            narration_id = entry.get('narration_id')
            noun_class = entry.get('noun_class')
            verb_class = entry.get('verb_class')
            
            if narration_id and noun_class is not None and verb_class is not None:
                fname = f"{narration_id}.mp4"
                rows.append({
                    'fname': fname,
                    'noun_id': int(noun_class),
                    'verb_id': int(verb_class)
                })
    
    # Write output CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fname', 'noun_id', 'verb_id'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Converted {len(rows)} entries from {input_csv_path} -> {output_csv_path}")
    
    # Print statistics
    if rows:
        noun_ids = set(r['noun_id'] for r in rows)
        verb_ids = set(r['verb_id'] for r in rows)
        print(f"  Unique nouns: {len(noun_ids)} (range: {min(noun_ids)}-{max(noun_ids)})")
        print(f"  Unique verbs: {len(verb_ids)} (range: {min(verb_ids)}-{max(verb_ids)})")
    
    return rows


def main():
    parser = argparse.ArgumentParser(description='Convert Epic Kitchens CSV to simplified format')
    parser.add_argument('--train_csv', type=str, default='EPIC_100_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='EPIC_100_validation.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for CSV files')
    parser.add_argument('--train_output', type=str, default='label.csv',
                        help='Output training CSV filename')
    parser.add_argument('--val_output', type=str, default='label.csv',
                        help='Output validation CSV filename')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_rows = []
    val_rows = []
    
    # Convert training set
    if os.path.exists(args.train_csv):
        # Put in train subdirectory
        train_dir = os.path.join(args.output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        train_output = os.path.join(train_dir, args.train_output)
        train_rows = convert_csv(args.train_csv, train_output)
    else:
        print(f"Warning: {args.train_csv} not found, skipping...")
    
    # Convert validation set
    if os.path.exists(args.val_csv):
        # Put in val subdirectory
        val_dir = os.path.join(args.output_dir, 'val')
        os.makedirs(val_dir, exist_ok=True)
        val_output = os.path.join(val_dir, args.val_output)
        val_rows = convert_csv(args.val_csv, val_output)
    else:
        print(f"Warning: {args.val_csv} not found, skipping...")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Training samples: {len(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")
    print(f"Total: {len(train_rows) + len(val_rows)}")
    print(f"\nOutput files:")
    if train_rows:
        print(f"  Train: {args.output_dir}/train/label.csv")
    if val_rows:
        print(f"  Val:   {args.output_dir}/val/label.csv")
    
    print(f"\n=== Expected Directory Structure ===")
    print(f"{args.output_dir}/")
    print(f"├── train/")
    print(f"│   ├── videos/")
    print(f"│   │   ├── P01_11_0.mp4")
    print(f"│   │   ├── P01_11_1.mp4")
    print(f"│   │   └── ...")
    print(f"│   └── label.csv")
    print(f"└── val/")
    print(f"    ├── videos/")
    print(f"    │   ├── P01_101_0.mp4")
    print(f"    │   └── ...")
    print(f"    └── label.csv")


if __name__ == '__main__':
    main()