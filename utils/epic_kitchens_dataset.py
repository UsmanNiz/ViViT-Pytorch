"""
Epic Kitchens Dataset Loader for ViViT Multi-Head Classification.

This module provides data loading utilities for the Epic Kitchens dataset,
supporting multi-head classification with verb and noun predictions.

Epic Kitchens dataset structure:
    - 97 verb classes
    - 300 noun classes
    - Labels are provided as concatenated one-hot vectors [verb, noun]
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

try:
    import decord
    from decord import cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Warning: decord not available. Video loading may be limited.")


class DecordInit:
    """Initialize video reader using Decord library."""
    
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = cpu(0) if DECORD_AVAILABLE else None
    
    def __call__(self, filename):
        if not DECORD_AVAILABLE:
            raise ImportError("decord is required for video loading")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Video file not found: {filename}")
        
        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError(f"Video file is empty: {filename}")
        
        reader = decord.VideoReader(
            filename,
            ctx=self.ctx,
            num_threads=self.num_threads
        )
        return reader


class EpicKitchensDataset(Dataset):
    """
    Epic Kitchens Dataset for multi-head noun/verb classification.
    
    Args:
        data_path: Path to video files directory
        annotations_path: Path to annotations CSV file
        num_frames: Number of frames to sample from each video
        tubelet_size: Size of temporal tubelets (default: 2)
        transform: Data augmentation transforms
        split: Dataset split - 'train', 'validation', or 'test'
        class_splits: List of number of classes per head [noun_classes, verb_classes]
                      Default: [300, 97] to match scenic-vivit order
        sample_method: Frame sampling method - 'tubelet' or 'uniform_sampling'
        one_hot_labels: Whether to return one-hot encoded labels
    """
    
    # Epic Kitchens class configuration
    NUM_NOUN_CLASSES = 300
    NUM_VERB_CLASSES = 97
    
    def __init__(
        self,
        data_path,
        annotations_path,
        num_frames=32,
        tubelet_size=2,
        transform=None,
        split='train',
        class_splits=None,
        sample_method='tubelet',
        one_hot_labels=True
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.transform = transform
        self.split = split
        self.sample_method = sample_method
        self.one_hot_labels = one_hot_labels
        
        # Set class splits (default order: noun, verb - matches scenic-vivit)
        if class_splits is None:
            self.class_splits = [self.NUM_NOUN_CLASSES, self.NUM_VERB_CLASSES]
        else:
            self.class_splits = class_splits
        
        self.total_classes = sum(self.class_splits)
        self.split_names = ['noun', 'verb']
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_path)
        
        # Video decoder
        self.v_decoder = DecordInit()
        
    def _load_annotations(self, annotations_path):
        """
        Load annotations from CSV file.
        
        Expected CSV format:
            - video_id or uid: unique identifier
            - participant_id: participant identifier
            - narration: text narration (optional)
            - start_frame: start frame number
            - stop_frame: stop frame number  
            - verb: verb label (string or class index)
            - verb_class: verb class index (0-96)
            - noun: noun label (string or class index)
            - noun_class: noun class index (0-299)
        
        Alternative format:
            - video_name/fname: video filename
            - verb_class/verb_label: verb class index
            - noun_class/noun_label: noun class index
        """
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        df = pd.read_csv(annotations_path)
        
        # Handle different column name conventions
        # Video ID column
        video_col = None
        for col in ['video_id', 'uid', 'fname', 'video_name', 'narration_id']:
            if col in df.columns:
                video_col = col
                break
        
        if video_col is None:
            raise ValueError("Could not find video identifier column in annotations")
        
        # Verb class column
        verb_col = None
        for col in ['verb_class', 'verb_label', 'verb']:
            if col in df.columns:
                verb_col = col
                break
        
        # Noun class column
        noun_col = None
        for col in ['noun_class', 'noun_label', 'noun']:
            if col in df.columns:
                noun_col = col
                break
        
        if verb_col is None or noun_col is None:
            raise ValueError("Could not find verb or noun class columns in annotations")
        
        # Rename columns for consistency
        df = df.rename(columns={
            video_col: 'video_id',
            verb_col: 'verb_class',
            noun_col: 'noun_class'
        })
        
        # Ensure class columns are integers
        df['verb_class'] = df['verb_class'].astype(int)
        df['noun_class'] = df['noun_class'].astype(int)
        
        return df
    
    def _get_video_path(self, video_id):
        """
        Construct the video path from video_id.
        
        Handles Epic Kitchens directory structure:
            data_path/P01/videos/P01_01.MP4
            data_path/P01/rgb_frames/P01_01/frame_0000000001.jpg
            
        Also handles alternative structures:
            data_path/P01/P01_01.mp4
            data_path/P01_01.mp4
        """
        video_id = str(video_id)
        
        # Extract participant_id from video_id (e.g., P01_01 -> P01)
        participant_id = None
        if '_' in video_id:
            parts = video_id.split('_')
            if len(parts) >= 2:
                participant_id = parts[0]  # P01, P02, etc.
        
        # Try different directory patterns for Epic Kitchens
        patterns = []
        
        if participant_id:
            # Epic Kitchens official structure: P01/videos/P01_01.MP4
            patterns.extend([
                os.path.join(participant_id, "videos", f"{video_id}.MP4"),
                os.path.join(participant_id, "videos", f"{video_id}.mp4"),
                os.path.join(participant_id, "videos", f"{video_id}.avi"),
                # Alternative: P01/P01_01.MP4
                os.path.join(participant_id, f"{video_id}.MP4"),
                os.path.join(participant_id, f"{video_id}.mp4"),
            ])
        
        # Flat structure fallback
        patterns.extend([
            f"{video_id}.MP4",
            f"{video_id}.mp4",
            f"{video_id}.avi",
            video_id,  # If video_id already includes extension
        ])
        
        for pattern in patterns:
            full_path = os.path.join(self.data_path, pattern)
            if os.path.exists(full_path):
                return full_path
        
        # Return default path (will raise error in loader if not found)
        if participant_id:
            return os.path.join(self.data_path, participant_id, "videos", f"{video_id}.MP4")
        return os.path.join(self.data_path, f"{video_id}.MP4")
    
    def _sample_frames(self, total_frames):
        """Sample frame indices from video."""
        sample_length = self.tubelet_size * self.num_frames
        
        if self.sample_method == 'tubelet':
            rand_end = max(0, total_frames - sample_length - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + sample_length, total_frames)
            frame_indices = np.linspace(begin_index, end_index - 1, self.num_frames, dtype=int)
        else:  # uniform_sampling
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        return frame_indices
    
    def _create_one_hot_labels(self, verb_class, noun_class):
        """
        Create concatenated one-hot labels for noun and verb.
        
        Order matches scenic-vivit: [noun, verb]
        
        Returns:
            Tensor of shape (total_classes,) with one-hot encoding
        """
        labels = torch.zeros(self.total_classes)
        
        # Noun one-hot (first class_splits[0] positions)
        if 0 <= noun_class < self.class_splits[0]:
            labels[noun_class] = 1.0
        
        # Verb one-hot (next class_splits[1] positions)
        verb_offset = self.class_splits[0]
        if 0 <= verb_class < self.class_splits[1]:
            labels[verb_offset + verb_class] = 1.0
        
        return labels
    
    def __getitem__(self, index):
        max_retries = 5
        
        for retry in range(max_retries):
            try:
                # Get annotation
                row = self.annotations.iloc[index]
                video_id = row['video_id']
                verb_class = int(row['verb_class'])
                noun_class = int(row['noun_class'])
                
                # Load video
                video_path = self._get_video_path(video_id)
                v_reader = self.v_decoder(video_path)
                total_frames = len(v_reader)
                
                # Sample frames
                frame_indices = self._sample_frames(total_frames)
                video = v_reader.get_batch(frame_indices).asnumpy()
                del v_reader
                
                # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
                with torch.no_grad():
                    video = torch.tensor(video, dtype=torch.float32)
                    video = video.permute(0, 3, 1, 2)
                    video = video / 255.0
                    
                    # Apply transforms
                    if self.transform is not None:
                        video = self.transform(video)
                
                # Create labels
                if self.one_hot_labels:
                    labels = self._create_one_hot_labels(verb_class, noun_class)
                else:
                    labels = {
                        'verb': torch.tensor(verb_class, dtype=torch.long),
                        'noun': torch.tensor(noun_class, dtype=torch.long)
                    }
                
                return video, labels
                
            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                if retry < max_retries - 1:
                    index = random.randint(0, len(self.annotations) - 1)
                else:
                    raise RuntimeError(f"Failed to load sample after {max_retries} retries")
    
    def __len__(self):
        return len(self.annotations)
    
    def get_class_weights(self):
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Dict with 'noun' and 'verb' weight tensors
        """
        noun_counts = np.zeros(self.class_splits[0])
        verb_counts = np.zeros(self.class_splits[1])
        
        for _, row in self.annotations.iterrows():
            verb_class = int(row['verb_class'])
            noun_class = int(row['noun_class'])
            
            if 0 <= noun_class < self.class_splits[0]:
                noun_counts[noun_class] += 1
            if 0 <= verb_class < self.class_splits[1]:
                verb_counts[verb_class] += 1
        
        # Avoid division by zero
        noun_counts = np.maximum(noun_counts, 1)
        verb_counts = np.maximum(verb_counts, 1)
        
        # Inverse frequency weighting
        noun_weights = 1.0 / noun_counts
        verb_weights = 1.0 / verb_counts
        
        # Normalize
        noun_weights = noun_weights / noun_weights.sum() * len(noun_weights)
        verb_weights = verb_weights / verb_weights.sum() * len(verb_weights)
        
        return {
            'noun': torch.tensor(noun_weights, dtype=torch.float32),
            'verb': torch.tensor(verb_weights, dtype=torch.float32)
        }


class EpicKitchensDatasetFromFrames(Dataset):
    """
    Epic Kitchens Dataset loading from pre-extracted frames.
    
    Use this class when frames have been pre-extracted from videos.
    
    Args:
        frames_path: Path to pre-extracted frames directory
        annotations_path: Path to annotations CSV file
        num_frames: Number of frames to sample
        transform: Data augmentation transforms
        split: Dataset split
        class_splits: Class configuration [noun_classes, verb_classes]
                      Default: [300, 97] to match scenic-vivit order
    """
    
    NUM_NOUN_CLASSES = 300
    NUM_VERB_CLASSES = 97
    
    def __init__(
        self,
        frames_path,
        annotations_path,
        num_frames=32,
        transform=None,
        split='train',
        class_splits=None,
        one_hot_labels=True
    ):
        self.frames_path = frames_path
        self.num_frames = num_frames
        self.transform = transform
        self.split = split
        self.one_hot_labels = one_hot_labels
        
        # Default order: noun, verb (matches scenic-vivit)
        if class_splits is None:
            self.class_splits = [self.NUM_NOUN_CLASSES, self.NUM_VERB_CLASSES]
        else:
            self.class_splits = class_splits
        
        self.total_classes = sum(self.class_splits)
        
        # Load annotations
        self.annotations = pd.read_csv(annotations_path)
        self._normalize_column_names()
    
    def _normalize_column_names(self):
        """Normalize column names to standard format."""
        column_mapping = {}
        
        for col in self.annotations.columns:
            col_lower = col.lower()
            if 'video' in col_lower or 'uid' in col_lower or 'fname' in col_lower:
                column_mapping[col] = 'video_id'
            elif 'verb' in col_lower and 'class' in col_lower:
                column_mapping[col] = 'verb_class'
            elif 'noun' in col_lower and 'class' in col_lower:
                column_mapping[col] = 'noun_class'
        
        self.annotations = self.annotations.rename(columns=column_mapping)
    
    def _get_frames_path(self, video_id):
        """
        Get the path to frames directory for a video.
        
        Handles Epic Kitchens structure:
            frames_path/P01/rgb_frames/P01_01/
            frames_path/P01/P01_01/
            frames_path/P01_01/
        """
        video_id = str(video_id)
        
        # Extract participant_id
        participant_id = None
        if '_' in video_id:
            parts = video_id.split('_')
            if len(parts) >= 2:
                participant_id = parts[0]
        
        # Try different patterns
        patterns = []
        if participant_id:
            patterns.extend([
                os.path.join(participant_id, "rgb_frames", video_id),
                os.path.join(participant_id, video_id),
            ])
        patterns.append(video_id)
        
        for pattern in patterns:
            full_path = os.path.join(self.frames_path, pattern)
            if os.path.isdir(full_path):
                return full_path
        
        # Return default
        if participant_id:
            return os.path.join(self.frames_path, participant_id, "rgb_frames", video_id)
        return os.path.join(self.frames_path, video_id)
    
    def _load_frames(self, video_id, frame_indices):
        """Load specific frames from disk."""
        from PIL import Image
        
        frames = []
        video_frames_path = self._get_frames_path(video_id)
        
        # Get list of available frames
        if os.path.isdir(video_frames_path):
            frame_files = sorted([f for f in os.listdir(video_frames_path) 
                                 if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])
        else:
            raise FileNotFoundError(f"Frame directory not found: {video_frames_path}")
        
        for idx in frame_indices:
            idx = min(idx, len(frame_files) - 1)
            frame_path = os.path.join(video_frames_path, frame_files[idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))
        
        return np.stack(frames)
    
    def _sample_frames(self, total_frames):
        """Sample frame indices."""
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return indices
    
    def _create_one_hot_labels(self, verb_class, noun_class):
        """Create concatenated one-hot labels (order: noun, verb)."""
        labels = torch.zeros(self.total_classes)
        
        # Noun first
        if 0 <= noun_class < self.class_splits[0]:
            labels[noun_class] = 1.0
        
        # Then verb
        verb_offset = self.class_splits[0]
        if 0 <= verb_class < self.class_splits[1]:
            labels[verb_offset + verb_class] = 1.0
        
        return labels
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        video_id = row['video_id']
        verb_class = int(row['verb_class'])
        noun_class = int(row['noun_class'])
        
        # Get frame count
        video_frames_path = self._get_frames_path(video_id)
        if os.path.isdir(video_frames_path):
            frame_files = [f for f in os.listdir(video_frames_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
            total_frames = len(frame_files)
        else:
            raise FileNotFoundError(f"Frame directory not found: {video_frames_path}")
        
        # Sample and load frames
        frame_indices = self._sample_frames(total_frames)
        video = self._load_frames(video_id, frame_indices)
        
        # Convert to tensor
        with torch.no_grad():
            video = torch.tensor(video, dtype=torch.float32)
            video = video.permute(0, 3, 1, 2)
            video = video / 255.0
            
            if self.transform is not None:
                video = self.transform(video)
        
        # Create labels
        if self.one_hot_labels:
            labels = self._create_one_hot_labels(verb_class, noun_class)
        else:
            labels = {
                'verb': torch.tensor(verb_class, dtype=torch.long),
                'noun': torch.tensor(noun_class, dtype=torch.long)
            }
        
        return video, labels
    
    def __len__(self):
        return len(self.annotations)

