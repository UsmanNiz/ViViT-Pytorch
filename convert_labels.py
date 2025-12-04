#!/usr/bin/env python3
"""
Convert label text files to CSV format expected by the ViViT dataloader.

Input format: /full/path/to/video.webm <label_number>
Output format: CSV with columns 'fname' and 'liveness_score'
"""

import pandas as pd
import os
import sys

def convert_label_file(input_txt_path, output_csv_path):
    """
    Convert a label text file to CSV format.
    
    Args:
        input_txt_path: Path to input text file (format: /path/to/video.webm label)
        output_csv_path: Path to output CSV file
    """
    data = []
    
    with open(input_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by space - last element is label, everything before is path
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            full_path, label = parts
            # Extract just the filename from the full path
            filename = os.path.basename(full_path)
            
            try:
                label = int(label)
            except ValueError:
                print(f"Warning: Invalid label '{label}' in line: {line}")
                continue
            
            data.append({
                'fname': filename,
                'liveness_score': label
            })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Converted {len(data)} entries from {input_txt_path} to {output_csv_path}")
    print(f"Sample of first 5 rows:")
    print(df.head())
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_labels.py <input_txt_file> <output_csv_file>")
        print("\nExample:")
        print("  python convert_labels.py train_sample.txt train_labels.csv")
        print("  python convert_labels.py val_sample.txt val_labels.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    convert_label_file(input_file, output_file)
    print(f"\n✓ Successfully created {output_file}")


