#!/usr/bin/env python3
"""
Preprocess Epic Kitchens dataset by trimming videos according to narration segments.

This script:
1. Reads Epic Kitchens annotations CSV
2. Trims videos based on start_frame and stop_frame for each narration
3. Saves trimmed videos with narration_id as filename
4. Creates a new CSV with: filename, path, verb_class, noun_class, and other necessary fields
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def get_video_fps(video_path):
    """Get FPS of a video file using ffprobe CLI."""
    try:
        # Use ffprobe CLI to get FPS
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 '{video_path}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 30.0
        else:
            fps = float(fps_str)
        return fps
    except:
        return 30.0  # Default FPS


def frame_to_timestamp(frame_num, fps):
    """Convert frame number to timestamp in seconds."""
    return frame_num / fps


def trim_video(input_path, output_path, start_frame, stop_frame, fps=None):
    """
    Trim video using ffmpeg CLI based on frame numbers.
    
    Args:
        input_path: Path to input video
        output_path: Path to output trimmed video
        start_frame: Start frame number
        stop_frame: Stop frame number
        fps: Frames per second (if None, will be detected using ffprobe)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    if fps is None:
        fps = get_video_fps(input_path)
    
    # Convert frames to timestamps
    start_time = frame_to_timestamp(start_frame, fps)
    duration = frame_to_timestamp(stop_frame - start_frame, fps)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build ffmpeg CLI command as a single string
    # Using frame-accurate trimming with -ss before -i for faster seeking
    # Optimized settings: ultrafast preset for speed, copy audio, no re-encoding if possible
    cmd = (
        f"ffmpeg -y -ss {start_time:.6f} -i '{input_path}' "
        f"-t {duration:.6f} -c:v libx264 -c:a copy "
        f"-preset ultrafast -crf 23 -movflags +faststart -loglevel error "
        f"'{output_path}'"
    )
    
    try:
        # Execute ffmpeg CLI command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout per video
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video {input_path}: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout trimming video {input_path}")
        return False


def find_source_video(video_id, data_dir):
    """
    Find the source video file for a given video_id.
    
    Returns:
        Path to source video file or None if not found
    """
    video_id = str(video_id)
    
    # Extract participant_id (e.g., P01_01 -> P01)
    participant_id = None
    if '_' in video_id:
        parts = video_id.split('_')
        if len(parts) >= 2:
            participant_id = parts[0]
    
    # Try different patterns
    patterns = []
    if participant_id:
        patterns.extend([
            os.path.join(data_dir, participant_id, "videos", f"{video_id}.MP4"),
            os.path.join(data_dir, participant_id, "videos", f"{video_id}.mp4"),
            os.path.join(data_dir, participant_id, "videos", f"{video_id}.avi"),
        ])
    
    patterns.extend([
        os.path.join(data_dir, f"{video_id}.MP4"),
        os.path.join(data_dir, f"{video_id}.mp4"),
    ])
    
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    
    return None


def process_video_clips(args):
    """
    Process all clips from a single source video - optimized worker function.
    This ensures all clips from one video are processed by the same thread,
    caching FPS and source video path lookups.
    
    Args:
        args: Tuple of (video_id, clips_list, source_video_dir, output_video_dir, skip_existing)
    
    Returns:
        List of result dictionaries (one per clip)
    """
    video_id, clips_list, source_video_dir, output_video_dir, skip_existing = args
    
    results = []
    
    # Find source video once for all clips
    source_video_path = find_source_video(video_id, source_video_dir)
    if source_video_path is None:
        # All clips from this video failed
        for row_dict in clips_list:
            results.append({
                'status': 'failed',
                'error': f'Source video not found for {video_id}',
                'narration_id': str(row_dict['narration_id'])
            })
        return results
    
    # Get FPS once for all clips (cache it for this video)
    try:
        fps = get_video_fps(source_video_path)
    except Exception as e:
        # All clips from this video failed
        for row_dict in clips_list:
            results.append({
                'status': 'failed',
                'error': f'Failed to get FPS: {str(e)}',
                'narration_id': str(row_dict['narration_id'])
            })
        return results
    
    # Process all clips from this video
    for row_dict in clips_list:
        narration_id = str(row_dict['narration_id'])
        start_frame = int(row_dict['start_frame'])
        stop_frame = int(row_dict['stop_frame'])
        verb_class = int(row_dict['verb_class'])
        noun_class = int(row_dict['noun_class'])
        
        # Output filename: narration_id.mp4
        output_filename = f"{narration_id}.mp4"
        output_path = os.path.join(output_video_dir, output_filename)
        
        # Skip if already exists
        if skip_existing and os.path.exists(output_path):
            results.append({
                'status': 'skipped',
                'narration_id': narration_id,
                'filename': output_filename,
                'path': output_path,
                'video_id': video_id,
                'verb_class': verb_class,
                'noun_class': noun_class,
                'verb': row_dict.get('verb', ''),
                'noun': row_dict.get('noun', ''),
                'narration': row_dict.get('narration', ''),
                'start_frame': start_frame,
                'stop_frame': stop_frame,
            })
            continue
        
        # Trim video (using cached FPS and source path)
        success = trim_video(source_video_path, output_path, start_frame, stop_frame, fps)
        
        if success:
            results.append({
                'status': 'success',
                'narration_id': narration_id,
                'filename': output_filename,
                'path': output_path,
                'video_id': video_id,
                'verb_class': verb_class,
                'noun_class': noun_class,
                'verb': row_dict.get('verb', ''),
                'noun': row_dict.get('noun', ''),
                'narration': row_dict.get('narration', ''),
                'start_frame': start_frame,
                'stop_frame': stop_frame,
            })
        else:
            results.append({
                'status': 'failed',
                'error': f'Failed to trim video',
                'narration_id': narration_id
            })
    
    return results


def preprocess_dataset(args):
    """
    Preprocess Epic Kitchens dataset with parallel processing.
    
    Args:
        args: argparse.Namespace with split parameter
    """
    # Hardcoded paths - update based on split
    split = args.split
    
    # Map split to annotation file names
    split_to_csv = {
        'train': 'EPIC_100_train.csv',
        'validation': 'EPIC_100_validation.csv',
        'test': 'EPIC_100_test_timestamps.csv'
    }
    
    # Base directories
    annotations_dir = 'annotations'
    source_video_dir = '/media/retrocausal-train/673ad668-0839-4e79-b61b-13759ab5c71f/Shaheer_and_Usman/epic-kitchens-dataset/EPIC-KITCHENS'
    output_video_base_dir = 'processed_videos'
    
    # Set paths based on split
    input_csv = os.path.join(annotations_dir, split_to_csv[split])
    output_video_dir = os.path.join(output_video_base_dir, split)
    
    print(f"Split: {split}")
    print(f"Reading annotations from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total samples in CSV: {len(df)}")
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Processing first {args.max_samples} samples")
    
    # Create output directory
    os.makedirs(output_video_dir, exist_ok=True)
    
    # Group clips by video_id for optimization (one thread per source video)
    print(f"\nGrouping clips by source video...")
    video_groups = df.groupby('video_id')
    video_clips = {}
    for video_id, group_df in video_groups:
        video_clips[video_id] = group_df.to_dict('records')
    
    print(f"Found {len(video_clips)} unique source videos")
    print(f"Total clips to process: {len(df)}")
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else min(cpu_count(), 16)
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for workers - each worker gets one video and all its clips
    worker_args = [
        (video_id, clips_list, source_video_dir, output_video_dir, not args.no_skip_existing)
        for video_id, clips_list in video_clips.items()
    ]
    
    # Process videos in parallel (one thread per source video)
    print(f"\nProcessing videos in parallel (one thread per source video)...")
    failed_count = 0
    skipped_count = 0
    processed_count = 0
    
    with Pool(processes=num_workers) as pool:
        # Each result is a list of results for all clips from one video
        video_results = list(tqdm(
            pool.imap(process_video_clips, worker_args),
            total=len(worker_args),
            desc="Processing videos"
        ))
    
    # Flatten results (each video returns a list of clip results)
    for video_result_list in video_results:
        for result in video_result_list:
            if result is None:
                failed_count += 1
                continue
            
            status = result.get('status', 'unknown')
            
            if status == 'success':
                processed_count += 1
            elif status == 'skipped':
                skipped_count += 1
            elif status == 'failed':
                failed_count += 1
                error = result.get('error', 'Unknown error')
                narration_id = result.get('narration_id', 'unknown')
                if failed_count <= 10:  # Only print first 10 errors
                    print(f"Failed to process {narration_id}: {error}")
    
    print(f"\n{'='*80}")
    print(f"Preprocessing Summary:")
    print(f"  Split: {split}")
    print(f"  Total samples: {len(df)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output videos directory: {output_video_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Epic Kitchens dataset by trimming videos"
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'validation', 'test'],
        help='Dataset split name (train/validation/test)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-process videos that already exist'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect, max 16)'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    # Run preprocessing
    preprocess_dataset(args)


if __name__ == '__main__':
    main()


