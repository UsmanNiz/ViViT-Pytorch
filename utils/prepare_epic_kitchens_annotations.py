"""
Prepare Epic Kitchens 100 annotations for ViViT training.

This script converts the official EK-100 annotations to the format
expected by our training pipeline.

Usage:
    python prepare_epic_kitchens_annotations.py \
        --input_csv path/to/EPIC_100_train.csv \
        --output_csv path/to/train_annotations.csv \
        --video_dir path/to/videos
"""

import argparse
import os
import pandas as pd


def prepare_annotations(input_csv, output_csv, video_dir=None, check_videos=False):
    """
    Prepare EK-100 annotations for training.
    
    Args:
        input_csv: Path to official EK-100 annotations CSV
        output_csv: Path to output processed annotations
        video_dir: Path to video directory (optional, for validation)
        check_videos: Whether to check if video files exist
    """
    print(f"Reading annotations from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Total samples: {len(df)}")
    
    # Create output dataframe with required columns
    output_df = pd.DataFrame()
    
    # Video ID - try different column names
    if 'narration_id' in df.columns:
        output_df['video_id'] = df['narration_id']
    elif 'uid' in df.columns:
        output_df['video_id'] = df['uid']
    elif 'video_id' in df.columns:
        output_df['video_id'] = df['video_id']
    else:
        raise ValueError("Could not find video identifier column")
    
    # Also keep the video_id for finding the video file
    if 'video_id' in df.columns:
        output_df['video_file'] = df['video_id']
    elif 'participant_id' in df.columns and 'video_id' in df.columns:
        # Format: P01_01
        output_df['video_file'] = df['video_id']
    else:
        # Try to extract from narration_id (e.g., P01_01_0 -> P01_01)
        output_df['video_file'] = output_df['video_id'].apply(
            lambda x: '_'.join(str(x).split('_')[:2]) if '_' in str(x) else str(x)
        )
    
    # Verb class
    if 'verb_class' in df.columns:
        output_df['verb_class'] = df['verb_class']
    else:
        raise ValueError("Could not find verb_class column")
    
    # Noun class
    if 'noun_class' in df.columns:
        output_df['noun_class'] = df['noun_class']
    else:
        raise ValueError("Could not find noun_class column")
    
    # Optional: keep additional useful columns
    optional_cols = ['start_frame', 'stop_frame', 'narration', 'verb', 'noun', 
                     'start_timestamp', 'stop_timestamp', 'participant_id']
    for col in optional_cols:
        if col in df.columns:
            output_df[col] = df[col]
    
    # Validate video files exist (optional)
    if check_videos and video_dir:
        print(f"\nValidating video files in: {video_dir}")
        missing = []
        unique_videos = output_df['video_file'].unique()
        print(f"Checking {len(unique_videos)} unique videos...")
        
        for video_file in unique_videos:
            # Extract participant ID (e.g., P01_01 -> P01)
            participant_id = video_file.split('_')[0] if '_' in video_file else None
            
            # Try different patterns for Epic Kitchens structure
            patterns = [
                # Official EK structure: P01/videos/P01_01.MP4
                os.path.join(video_dir, participant_id, "videos", f"{video_file}.MP4") if participant_id else None,
                os.path.join(video_dir, participant_id, "videos", f"{video_file}.mp4") if participant_id else None,
                # Alternative: P01/P01_01.MP4
                os.path.join(video_dir, participant_id, f"{video_file}.MP4") if participant_id else None,
                os.path.join(video_dir, participant_id, f"{video_file}.mp4") if participant_id else None,
                # Flat structure
                os.path.join(video_dir, f"{video_file}.MP4"),
                os.path.join(video_dir, f"{video_file}.mp4"),
            ]
            patterns = [p for p in patterns if p is not None]
            found = any(os.path.exists(p) for p in patterns)
            if not found:
                missing.append(video_file)
        
        if missing:
            print(f"Warning: {len(missing)} videos not found!")
            print(f"First 10 missing: {missing[:10]}")
            print(f"\nExpected structure: {video_dir}/P01/videos/P01_01.MP4")
        else:
            print("All videos found!")
    
    # Print statistics
    print(f"\nProcessed annotations:")
    print(f"  Total samples: {len(output_df)}")
    print(f"  Unique videos: {output_df['video_file'].nunique()}")
    print(f"  Verb classes: {output_df['verb_class'].nunique()} (range: {output_df['verb_class'].min()}-{output_df['verb_class'].max()})")
    print(f"  Noun classes: {output_df['noun_class'].nunique()} (range: {output_df['noun_class'].min()}-{output_df['noun_class'].max()})")
    
    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Prepare EK-100 annotations")
    parser.add_argument("--input_csv", required=True, help="Path to official EK-100 CSV")
    parser.add_argument("--output_csv", required=True, help="Output processed CSV path")
    parser.add_argument("--video_dir", default=None, help="Video directory for validation")
    parser.add_argument("--check_videos", action='store_true', help="Check if videos exist")
    
    args = parser.parse_args()
    prepare_annotations(args.input_csv, args.output_csv, args.video_dir, args.check_videos)


if __name__ == "__main__":
    main()

