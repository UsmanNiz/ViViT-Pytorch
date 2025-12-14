# coding=utf-8
"""
Data loading utilities for ViViT training with full augmentation support.

Builds on existing CustomDataset and adds ViViT paper augmentations:
- Scale jitter (0.9-1.33)
- Color jitter (for K400/K600/MiT)
- RandAugment (for EK/SSv2)

Reference: ViViT paper Table 7 (https://arxiv.org/abs/2103.15691)
"""
import logging
import os
import subprocess
import re
import random

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

from utils.custom_dataset import CustomDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Video Transform Classes (Temporally Consistent)
# =============================================================================

class VideoTransformWrapper(object):
    """Wrapper to apply image transforms to video tensors (T, C, H, W)"""
    def __init__(self, transform):
        self.transform = transform
        self.transform_name = transform.__class__.__name__
    
    def __call__(self, x):
        if x.dim() == 4 and x.shape[0] > 1:
            transformed_frames = []
            for t in range(x.shape[0]):
                frame_result = self.transform(x[t])
                transformed_frames.append(frame_result)
            return torch.stack(transformed_frames, dim=0)
        else:
            return self.transform(x)


class VideoRandomResizedCrop:
    """
    Random resized crop with scale jitter - same crop for all frames.
    From ViViT paper: scale=(0.9, 1.33)
    """
    def __init__(self, size, scale=(0.9, 1.33), ratio=(0.75, 1.33)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, x):
        if x.dim() == 4:  # (T, C, H, W)
            # Get random crop parameters once for all frames
            i, j, h, w = T.RandomResizedCrop.get_params(
                x[0], scale=self.scale, ratio=self.ratio
            )
            cropped_frames = []
            for t in range(x.shape[0]):
                cropped = TF.resized_crop(x[t], i, j, h, w, self.size)
                cropped_frames.append(cropped)
            return torch.stack(cropped_frames, dim=0)
        else:
            i, j, h, w = T.RandomResizedCrop.get_params(
                x, scale=self.scale, ratio=self.ratio
            )
            return TF.resized_crop(x, i, j, h, w, self.size)


class VideoRandomHorizontalFlip:
    """Random horizontal flip - same decision for all frames."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            if x.dim() == 4:
                return torch.flip(x, dims=[-1])
            else:
                return TF.hflip(x)
        return x


class VideoColorJitter:
    """
    Color jitter - same parameters for all frames.
    From ViViT paper: p=0.8, used for K400/K600/MiT, NOT for EK/SSv2.
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        
        # Get random parameters once
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            T.ColorJitter.get_params(
                brightness=(max(0, 1 - self.brightness), 1 + self.brightness),
                contrast=(max(0, 1 - self.contrast), 1 + self.contrast),
                saturation=(max(0, 1 - self.saturation), 1 + self.saturation),
                hue=(-self.hue, self.hue)
            )
        
        def apply_jitter(frame):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    frame = TF.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = TF.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = TF.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = TF.adjust_hue(frame, hue_factor)
            return frame
        
        if x.dim() == 4:
            return torch.stack([apply_jitter(x[t]) for t in range(x.shape[0])], dim=0)
        else:
            return apply_jitter(x)


class VideoRandAugment:
    """
    RandAugment for video - same augmentation for all frames.
    From ViViT paper: EK: layers=2, mag=15; SSv2: layers=2, mag=20
    """
    OPERATIONS = [
        'identity', 'autocontrast', 'equalize', 'rotate', 'solarize',
        'posterize', 'contrast', 'brightness', 'sharpness',
        'shear_x', 'shear_y', 'translate_x', 'translate_y'
    ]
    
    def __init__(self, num_layers=2, magnitude=15):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_magnitude = 30
    
    def _get_magnitude(self, name):
        m = self.magnitude / self.max_magnitude
        magnitudes = {
            'rotate': 30.0 * m,
            'solarize': 1.0 - m,
            'posterize': max(1, int(4 - 4 * m)),
            'contrast': 0.9 * m,
            'brightness': 0.9 * m,
            'sharpness': 0.9 * m,
            'shear_x': 0.3 * m,
            'shear_y': 0.3 * m,
            'translate_x': 0.45 * m,
            'translate_y': 0.45 * m,
        }
        return magnitudes.get(name, 0)
    
    def _apply_op(self, frame, op, sign=1):
        mag = self._get_magnitude(op)
        
        if op == 'identity':
            return frame
        elif op == 'autocontrast':
            return TF.autocontrast(frame)
        elif op == 'equalize':
            frame_uint8 = (frame * 255).clamp(0, 255).to(torch.uint8)
            return TF.equalize(frame_uint8).float() / 255.0
        elif op == 'rotate':
            return TF.rotate(frame, sign * mag)
        elif op == 'solarize':
            return TF.solarize(frame, mag)
        elif op == 'posterize':
            frame_uint8 = (frame * 255).clamp(0, 255).to(torch.uint8)
            return TF.posterize(frame_uint8, int(mag)).float() / 255.0
        elif op == 'contrast':
            return TF.adjust_contrast(frame, 1 + sign * mag)
        elif op == 'brightness':
            return TF.adjust_brightness(frame, 1 + sign * mag)
        elif op == 'sharpness':
            return TF.adjust_sharpness(frame, 1 + sign * mag)
        elif op == 'shear_x':
            return TF.affine(frame, 0, [0, 0], 1.0, [sign * mag * 57.3, 0])
        elif op == 'shear_y':
            return TF.affine(frame, 0, [0, 0], 1.0, [0, sign * mag * 57.3])
        elif op == 'translate_x':
            w = frame.shape[-1]
            return TF.affine(frame, 0, [int(sign * mag * w), 0], 1.0, [0, 0])
        elif op == 'translate_y':
            h = frame.shape[-2]
            return TF.affine(frame, 0, [0, int(sign * mag * h)], 1.0, [0, 0])
        return frame
    
    def __call__(self, x):
        # Select operations and signs once for entire video
        ops = random.choices(self.OPERATIONS, k=self.num_layers)
        signs = [random.choice([-1, 1]) for _ in range(self.num_layers)]
        
        if x.dim() == 4:
            frames = []
            for t in range(x.shape[0]):
                frame = x[t]
                for op, sign in zip(ops, signs):
                    frame = self._apply_op(frame, op, sign)
                frames.append(frame)
            return torch.stack(frames, dim=0)
        else:
            for op, sign in zip(ops, signs):
                x = self._apply_op(x, op, sign)
            return x


class VideoNormalize:
    """Normalize video with ImageNet statistics."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, x):
        if x.dim() == 4:
            return (x - self.mean) / self.std
        else:
            return (x - self.mean.squeeze(0)) / self.std.squeeze(0)


class MyRotateTransform(object):
    """Your existing rotation transform."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles, p=[0.8, 0.2])
        angle = float(angle) if isinstance(angle, (np.integer, np.floating)) else angle
        
        original_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        
        if x.dim() == 4 and x.shape[0] > 1:
            rotated_frames = [TF.rotate(x[t], angle) for t in range(x.shape[0])]
            result = torch.stack(rotated_frames, dim=0)
        else:
            result = TF.rotate(x, angle)
        
        if original_dtype == torch.float16:
            result = result.half()
        
        return result


# =============================================================================
# Transform Presets
# =============================================================================

def get_vivit_transforms(dataset_name='custom', img_size=224, is_training=True):
    """
    Get transforms matching ViViT paper Table 7.
    
    Args:
        dataset_name: 'kinetics400', 'epic_kitchens', 'ssv2', 'custom', etc.
        img_size: Target size (default 224)
        is_training: Training or validation transforms
    """
    if is_training:
        transforms_list = [
            # Scale jitter (paper: 0.9-1.33 for all datasets)
            VideoRandomResizedCrop(img_size, scale=(0.9, 1.33)),
            # Random horizontal flip (paper: p=0.5)
            VideoRandomHorizontalFlip(p=0.5),
        ]
        
        # Dataset-specific augmentations
        if dataset_name in ['kinetics400', 'kinetics600', 'moments_in_time']:
            # Color jitter (paper: p=0.8)
            transforms_list.append(VideoColorJitter(p=0.8))
        elif dataset_name == 'epic_kitchens':
            # RandAugment (paper: layers=2, magnitude=15)
            transforms_list.append(VideoRandAugment(num_layers=2, magnitude=15))
        elif dataset_name == 'ssv2':
            # RandAugment (paper: layers=2, magnitude=20)
            transforms_list.append(VideoRandAugment(num_layers=2, magnitude=20))
        else:
            # Custom/default: use color jitter
            transforms_list.append(VideoColorJitter(p=0.8))
        
        transforms_list.append(VideoNormalize())
    else:
        # Validation transforms
        transforms_list = [
            VideoTransformWrapper(T.Resize((img_size, img_size))),
            VideoNormalize(),
        ]
    
    return T.Compose(transforms_list)


# Keep your existing transforms for backward compatibility
data_transforms = {
    'train': T.Compose([
        VideoTransformWrapper(T.Resize((256, 256))),
        VideoTransformWrapper(T.RandomCrop((224, 224))),
        VideoTransformWrapper(T.RandomHorizontalFlip()),
        MyRotateTransform([0, 180]),
        VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ]),
    'val': T.Compose([
        VideoTransformWrapper(T.Resize((224, 224))),
        VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ]),
    'test': T.Compose([
        VideoTransformWrapper(T.Resize((224, 224))),
        VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ])
}


# =============================================================================
# Dataset Configurations (ViViT Paper Table 7)
# =============================================================================

DATASET_CONFIGS = {
    'kinetics400': {
        'num_classes': 400,
        'learning_rate': 0.1,
        'epochs': 30,
        'warmup_epochs': 2.5,
        'label_smoothing': 0.0,
        'mixup_alpha': 0.0,
        'stochastic_depth': 0.0,
    },
    'kinetics600': {
        'num_classes': 600,
        'learning_rate': 0.1,
        'epochs': 30,
        'warmup_epochs': 2.5,
        'label_smoothing': 0.0,
        'mixup_alpha': 0.0,
        'stochastic_depth': 0.0,
    },
    'moments_in_time': {
        'num_classes': 339,
        'learning_rate': 0.25,
        'epochs': 10,
        'warmup_epochs': 2.5,
        'label_smoothing': 0.0,
        'mixup_alpha': 0.0,
        'stochastic_depth': 0.0,
    },
    'epic_kitchens': {
        'num_classes': 397,
        'class_splits': [300, 97],
        'learning_rate': 0.5,
        'epochs': 50,
        'warmup_epochs': 2.5,
        'label_smoothing': 0.2,
        'mixup_alpha': 0.1,
        'stochastic_depth': 0.2,
    },
    'ssv2': {
        'num_classes': 174,
        'learning_rate': 0.5,
        'epochs': 35,
        'warmup_epochs': 2.5,
        'label_smoothing': 0.3,
        'mixup_alpha': 0.3,
        'stochastic_depth': 0.3,
    },
}


# =============================================================================
# Main Data Loader Function
# =============================================================================

def get_loader(args):
    """
    Get data loaders with appropriate transforms.
    
    Supports:
    - CIFAR10/CIFAR100 (for testing)
    - Custom video datasets
    - ViViT paper augmentations when using --use_vivit_aug flag
    """
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Image transforms for CIFAR
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    
    else:
        # Video datasets
        # Choose transforms based on dataset and flags
        use_vivit_aug = getattr(args, 'use_vivit_aug', False) or args.dataset in DATASET_CONFIGS
        
        if use_vivit_aug:
            train_transform = get_vivit_transforms(args.dataset, args.img_size, is_training=True)
            val_transform = get_vivit_transforms(args.dataset, args.img_size, is_training=False)
            logger.info(f"Using ViViT paper augmentations for {args.dataset}")
        else:
            train_transform = data_transforms["train"]
            val_transform = data_transforms["val"]
            logger.info("Using default augmentations")
        
        # Helper to determine video directory path
        def get_video_dir(base_dir):
            """Check if videos/ subfolder exists, otherwise use base_dir directly."""
            video_subdir = os.path.join(base_dir, "videos")
            if os.path.isdir(video_subdir):
                return video_subdir
            return base_dir
        
        # Get class_splits for Epic Kitchens
        class_splits = getattr(args, 'class_splits', None)
        
        if args.data_dir and not args.test_dir:
            # Single directory with train/val split
            data = CustomDataset(
                get_video_dir(args.data_dir),
                args.data_dir + "/label.csv",
                args.num_frames,
                transform=train_transform,
                blackbar_check=None,
                class_splits=class_splits
            )
            try:
                trainset, testset = random_split(
                    data, [0.8, 0.2], 
                    generator=torch.Generator().manual_seed(args.seed)
                )
            except:
                train_size = int(0.8 * len(data))
                test_size = len(data) - train_size
                trainset, testset = random_split(
                    data, [train_size, test_size],
                    generator=torch.Generator().manual_seed(args.seed)
                )
            testset.dataset.set_transform(val_transform)
            
        elif args.data_dir and args.test_dir:
            # Separate train and test directories
            trainset = CustomDataset(
                get_video_dir(args.data_dir),
                args.data_dir + "/label.csv",
                args.num_frames,
                transform=train_transform,
                blackbar_check=None,
                class_splits=class_splits
            )
            testset = CustomDataset(
                get_video_dir(args.test_dir),
                args.test_dir + "/label.csv",
                args.num_frames,
                transform=val_transform,
                blackbar_check=None,
                class_splits=class_splits
            )
            
        elif not args.data_dir and args.test_dir:
            # Test only
            testset = CustomDataset(
                get_video_dir(args.test_dir),
                None,
                args.num_frames,
                transform=val_transform,
                blackbar_check=None,
                class_splits=class_splits
            )
            trainset = None
        
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    # Create data loaders
    num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers > 0 else min(8, os.cpu_count() or 1)
    
    if trainset is not None:
        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        train_loader = DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
    else:
        train_loader = None
        
    if testset is not None:
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
    else:
        test_loader = None

    return train_loader, test_loader


# =============================================================================
# Utility Functions
# =============================================================================

def get_blackbar(vid_path):
    """Detect black bars in video for cropping."""
    CROP_DETECT_LINE = b'w:(\d+)\sh:(\d+)\sx:(\d+)\sy:(\d+)'
    CROP_COORDINATE = b'x1:(\d+)\sx2:(\d+)\sy1:(\d+)\sy2:(\d+)'
    p = subprocess.Popen(
        ["ffmpeg", "-i", vid_path, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    infos = p.stderr.read()
    crop_coordinate = re.findall(CROP_COORDINATE, infos)
    crop_data = re.findall(CROP_DETECT_LINE, infos)
    
    if not crop_coordinate:
        return None
    
    crop_coordinate = crop_coordinate[0]
    if int(crop_coordinate[0].decode('utf8')) == 0 and int(crop_coordinate[2].decode('utf8')) == 0:
        return None
    else:
        output = [int(crop.decode('utf8')) for crop in crop_data[0]]
    
    return output


def get_dataset_config(dataset_name):
    """Get default configuration for a dataset from ViViT paper."""
    return DATASET_CONFIGS.get(dataset_name, {})