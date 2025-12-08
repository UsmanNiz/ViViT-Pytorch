import logging
import os

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split
import subprocess
import re
from utils.custom_dataset import CustomDataset
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class VideoTransformWrapper(object):
    """Wrapper to apply image transforms to video tensors (T, C, H, W)"""
    def __init__(self, transform):
        self.transform = transform
        self.transform_name = transform.__class__.__name__
    
    def __call__(self, x):
        # print(f"[DEBUG VideoTransformWrapper] Applying {self.transform_name}, input shape: {x.shape}")
        # If it's a video tensor (T, C, H, W) with T > 1
        if x.dim() == 4 and x.shape[0] > 1:
            # Apply transform to each frame
            transformed_frames = []
            for t in range(x.shape[0]):
                frame_result = self.transform(x[t])
                transformed_frames.append(frame_result)
            result = torch.stack(transformed_frames, dim=0)
            # print(f"[DEBUG VideoTransformWrapper] Final video shape: {result.shape}")
            return result
        else:
            # Single frame or image
            result = self.transform(x)
            # print(f"[DEBUG VideoTransformWrapper] Single frame result shape: {result.shape}")
            return result


class MyRotateTransform(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles, p=[0.8, 0.2])
        # Convert numpy scalar to Python float/int (required by torchvision)
        angle = float(angle) if isinstance(angle, (np.integer, np.floating)) else angle
        # print(f"[DEBUG MyRotateTransform] Rotating video by {angle} degrees, input shape: {x.shape}, dtype: {x.dtype}")
        
        # Convert to float32 if needed (rotation doesn't support float16 on CPU)
        original_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
            # print(f"[DEBUG MyRotateTransform] Converted to float32 for rotation")
        
        # Handle video tensors: (T, C, H, W) - apply rotation to each frame
        if x.dim() == 4 and x.shape[0] > 1:  # Video tensor (T, C, H, W)
            # print(f"[DEBUG MyRotateTransform] Processing video tensor with {x.shape[0]} frames")
            # Apply rotation to each frame
            rotated_frames = []
            for t in range(x.shape[0]):
                rotated_frames.append(F.rotate(x[t], angle))
            result = torch.stack(rotated_frames, dim=0)
        else:
            # Single frame or image
            # print(f"[DEBUG MyRotateTransform] Processing single frame/image")
            result = F.rotate(x, angle)
        
        # Convert back to original dtype if needed
        if original_dtype == torch.float16:
            result = result.half()
            # print(f"[DEBUG MyRotateTransform] Converted back to float16")
        
        # print(f"[DEBUG MyRotateTransform] Output shape: {result.shape}, dtype: {result.dtype}")
        return result
    


data_transforms = {

'train': T.Compose([
	VideoTransformWrapper(T.Resize((256, 256))),  # Resize first to ensure minimum size
	VideoTransformWrapper(T.RandomCrop((224, 224))),  # Then crop to exact size
	VideoTransformWrapper(T.RandomHorizontalFlip()),
	MyRotateTransform([0, 180]),
	VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
]),

'val': T.Compose([
	VideoTransformWrapper(T.Resize((224, 224))),  # Use Resize instead of RandomResizedCrop for consistent sizing
	VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
]),

'test': T.Compose([
	VideoTransformWrapper(T.Resize((224, 224))),  # Use Resize instead of RandomResizedCrop for consistent sizing
	VideoTransformWrapper(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
])
}



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

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
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    
    else:
    
        if args.data_dir and not args.test_dir:
            data = CustomDataset(args.data_dir + "/videos",args.data_dir + "/label.csv",args.num_frames,transform=data_transforms["train"], blackbar_check=None)
            try:
                trainset,testset = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(args.seed))
            except:
                trainset,testset = random_split(data, [935, 233], generator=torch.Generator().manual_seed(args.seed))
            testset.dataset.set_transform(data_transforms["val"])
        elif args.data_dir and args.test_dir:
            trainset = CustomDataset(args.data_dir + "/videos",args.data_dir + "/label.csv",args.num_frames,transform=data_transforms["train"], blackbar_check=None)
            testset = CustomDataset(args.test_dir + "/videos",args.test_dir + "/label.csv",args.num_frames,transform=data_transforms["test"], blackbar_check=None)
        elif not args.data_dir and args.test_dir:
            testset = CustomDataset(args.test_dir + "/videos",None,args.num_frames,transform=data_transforms["test"], blackbar_check=None)
            trainset = None
        
    if args.local_rank == 0:
        torch.distributed.barrier()
    if trainset is not None:
        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        # Use multiple workers for data loading to prevent GPU starvation
        # Set num_workers based on available CPUs, but cap at reasonable number
        num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers > 0 else min(8, os.cpu_count() or 1)
        train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True if num_workers > 0 else False,
                              prefetch_factor=2 if num_workers > 0 else None)
    else:
        train_loader = None
    if testset is not None:
        test_sampler = SequentialSampler(testset)
        # Use multiple workers for data loading to prevent GPU starvation
        num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers > 0 else min(4, os.cpu_count() or 1)
        test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             persistent_workers=True if num_workers > 0 else False,
                             prefetch_factor=2 if num_workers > 0 else None)
    else:
        train_loader = None
        
    

    return train_loader, test_loader

def get_blackbar(vid_path):
    CROP_DETECT_LINE = b'w:(\d+)\sh:(\d+)\sx:(\d+)\sy:(\d+)'
    CROP_COORDINATE = b'x1:(\d+)\sx2:(\d+)\sy1:(\d+)\sy2:(\d+)'
    p = subprocess.Popen(["ffmpeg", "-i", vid_path, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"]
                    , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    infos = p.stderr.read()
    crop_coordinate = re.findall(CROP_COORDINATE , infos) #y1,y2,x1,x2
    crop_data = re.findall(CROP_DETECT_LINE , infos) #(width,height,left,top)
    crop_coordinate = crop_coordinate[0]
    if int(crop_coordinate[0].decode('utf8')) == 0 and int(crop_coordinate[2].decode('utf8')) == 0:
        return None
    else:
        output = [int(crop.decode('utf8')) for crop in crop_data[0]] 
    
    return output



