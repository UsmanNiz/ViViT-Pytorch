# coding=utf-8
"""
ViViT Training Script for Epic Kitchens Multi-Head Classification.

This script supports training ViViT models with multiple classification heads
for predicting both verbs and nouns in the Epic Kitchens dataset.

Features:
- Multi-head classification (verb + noun)
- Native PyTorch AMP (FP16)
- Distributed training support
- Per-head accuracy tracking
- Joint accuracy metrics
- Checkpoint resume capability
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta, datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import CONFIGS, ViViTMultiHead

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_epic_kitchens_loader
from utils.regularization import Mixup, mixup_criterion, EMA


logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiHeadMetrics:
    """Track metrics for multi-head classification."""
    
    def __init__(self, class_splits, split_names):
        self.class_splits = class_splits
        self.split_names = split_names
        self.cumulative_splits = np.cumsum([0] + class_splits).tolist()
        self.reset()
    
    def reset(self):
        self.all_preds = {name: [] for name in self.split_names}
        self.all_labels = {name: [] for name in self.split_names}
        self.all_logits = {name: [] for name in self.split_names}
    
    def update(self, logits, labels):
        """
        Update metrics with batch predictions.
        
        Args:
            logits: Concatenated logits (B, total_classes)
            labels: One-hot concatenated labels (B, total_classes)
        """
        # Split logits and labels per head
        logits_splits = torch.split(logits, self.class_splits, dim=-1)
        labels_splits = torch.split(labels, self.class_splits, dim=-1)
        
        for i, name in enumerate(self.split_names):
            head_logits = logits_splits[i].detach().cpu().numpy()
            head_labels = labels_splits[i].detach().cpu().numpy()
            
            # Get predictions and ground truth
            preds = np.argmax(head_logits, axis=-1)
            gt = np.argmax(head_labels, axis=-1)
            
            self.all_preds[name].extend(preds.tolist())
            self.all_labels[name].extend(gt.tolist())
            self.all_logits[name].extend(head_logits.tolist())
    
    def compute_accuracy(self, name):
        """Compute top-1 accuracy for a specific head."""
        preds = np.array(self.all_preds[name])
        labels = np.array(self.all_labels[name])
        return (preds == labels).mean()
    
    def compute_top_k_accuracy(self, name, k=5):
        """Compute top-k accuracy for a specific head."""
        logits = np.array(self.all_logits[name])
        labels = np.array(self.all_labels[name])
        
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i]:
                correct += 1
        return correct / len(labels)
    
    def compute_joint_accuracy(self):
        """Compute joint accuracy (both heads correct)."""
        correct = 0
        total = len(self.all_preds[self.split_names[0]])
        
        for i in range(total):
            all_correct = True
            for name in self.split_names:
                if self.all_preds[name][i] != self.all_labels[name][i]:
                    all_correct = False
                    break
            if all_correct:
                correct += 1
        
        return correct / total if total > 0 else 0
    
    def get_all_metrics(self):
        """Get all metrics as a dictionary."""
        metrics = {}
        
        for name in self.split_names:
            metrics[f'{name}_top1'] = self.compute_accuracy(name)
            num_classes = self.class_splits[self.split_names.index(name)]
            k = min(5, num_classes)
            metrics[f'{name}_top{k}'] = self.compute_top_k_accuracy(name, k=k)
        
        metrics['joint_accuracy'] = self.compute_joint_accuracy()
        
        return metrics


class TextLogger:
    """Text file logger for training metrics."""
    
    def __init__(self, log_path, args, class_splits, split_names):
        self.log_path = log_path
        self.file = open(log_path, "a")
        self.split_names = split_names
        
        # Write header
        self.file.write("=" * 120 + "\n")
        self.file.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"Experiment: {args.name}\n")
        self.file.write("-" * 120 + "\n")
        self.file.write(f"Model: {args.model_type}\n")
        self.file.write(f"Dataset: Epic Kitchens\n")
        self.file.write(f"Class splits: {class_splits} ({split_names})\n")
        self.file.write(f"Total classes: {sum(class_splits)}\n")
        self.file.write(f"Num frames: {args.num_frames}\n")
        self.file.write(f"Batch size: {args.train_batch_size}\n")
        self.file.write(f"Learning rate: {args.learning_rate}\n")
        self.file.write(f"Label smoothing: {args.label_smoothing}\n")
        self.file.write(f"FP16: {args.fp16}\n")
        self.file.write("=" * 120 + "\n\n")
        
        # Write column headers
        headers = ['Step', 'Epoch', 'Loss']
        for name in split_names:
            headers.extend([f'{name}_Acc', f'{name}_Top5'])
        headers.extend(['Joint_Acc', 'LR', 'Note'])
        
        header_str = ''.join([f'{h:<12}' for h in headers])
        self.file.write(header_str + "\n")
        self.file.write("-" * 120 + "\n")
        self.file.flush()
    
    def log(self, step, epoch, loss, metrics, lr, note=""):
        values = [f'{step:<12}', f'{epoch:<12}', f'{loss:<12.5f}']
        
        for name in self.split_names:
            values.append(f'{metrics.get(f"{name}_top1", 0):<12.5f}')
            # Handle different top-k keys
            top5_key = f'{name}_top5' if f'{name}_top5' in metrics else f'{name}_top{min(5, 97 if name == "verb" else 300)}'
            values.append(f'{metrics.get(top5_key, 0):<12.5f}')
        
        values.extend([f'{metrics.get("joint_accuracy", 0):<12.5f}', f'{lr:<12.8f}', note])
        
        self.file.write(''.join(values) + "\n")
        self.file.flush()
    
    def log_message(self, message):
        self.file.write(f">>> {message}\n")
        self.file.flush()
    
    def close(self, best_metrics):
        self.file.write("\n" + "=" * 120 + "\n")
        self.file.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write("Best Metrics:\n")
        for key, value in best_metrics.items():
            self.file.write(f"  {key}: {value:.5f}\n")
        self.file.write("=" * 120 + "\n")
        self.file.close()


def save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, 
               best_metrics, config, best=False, step_checkpoint=False, ema=None):
    """Save model checkpoint."""
    model_to_save = model.module if hasattr(model, 'module') else model
    
    if best:
        checkpoint_path = os.path.join(args.output_dir, f"{args.name}_best.pth")
    elif step_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, f"{args.name}_step_{global_step}.pth")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"{args.name}_epoch_{epoch}.pth")
    
    checkpoint = {
        'config': config,
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'best_metrics': best_metrics,
        'args': args,
        'class_splits': model_to_save.class_splits,
        'split_names': model_to_save.split_names,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(args, model, optimizer=None, scheduler=None, scaler=None, ema=None):
    """Load checkpoint for resuming training."""
    logger.info(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    if ema is not None and checkpoint.get('ema_state_dict') is not None:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("Loaded EMA state")
    
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_metrics = checkpoint.get('best_metrics', {})
    
    logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    return start_epoch, global_step, best_metrics


def setup(args):
    """Set up model and configuration."""
    config = CONFIGS[args.model_type]
    
    # Override label smoothing from args if provided
    if hasattr(args, 'label_smoothing'):
        config.label_smoothing = args.label_smoothing
    
    # Override stochastic droplayer rate from args
    if hasattr(args, 'stochastic_droplayer_rate'):
        config.stochastic_droplayer_rate = args.stochastic_droplayer_rate
    
    # Get class splits from config
    class_splits = config.class_splits if hasattr(config, 'class_splits') else [97, 300]
    split_names = config.split_names if hasattr(config, 'split_names') else ['verb', 'noun']
    
    # Update args with class splits for data loading
    args.class_splits = class_splits
    args.split_names = split_names
    
    # Create model
    model = ViViTMultiHead(
        config,
        image_size=args.img_size,
        class_splits=class_splits,
        split_names=split_names,
        num_frames=args.num_frames,
        pool='cls'
    )
    
    # Load pretrained weights if not resuming
    if args.resume is None and args.pretrained_dir:
        logger.info(f"Loading pretrained weights from {args.pretrained_dir}")
        model.load_from(np.load(args.pretrained_dir))
    
    model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Class splits: {class_splits} ({split_names})")
    logger.info(f"Total parameters: {num_params:.1f}M")
    
    return args, model, config


def set_seed(args):
    """Set random seeds for reproducibility."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def validate(args, model, test_loader, epoch, global_step, writer=None, full_eval=True, scaler=None):
    """
    Run validation with multi-head metrics.
    
    Returns:
        Dict of metrics, validation loss
    """
    eval_losses = AverageMeter()
    metrics_tracker = MultiHeadMetrics(args.class_splits, args.split_names)
    
    if full_eval:
        logger.info("***** Running Validation *****")
        logger.info(f"  Num batches = {len(test_loader)}")
    
    model.eval()
    
    max_batches = len(test_loader) if full_eval else min(10, len(test_loader))
    
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating...",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0] or not full_eval
    )
    
    for step, batch in enumerate(epoch_iterator):
        if step >= max_batches:
            break
        
        videos, labels = batch
        videos = videos.to(args.device)
        labels = labels.to(args.device)
        
        # Handle video dimension
        if videos.dim() == 4:
            videos = videos.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        
        with torch.no_grad():
            if args.fp16:
                with autocast():
                    loss = model(videos, labels)
                    logits = model(videos)
            else:
                loss = model(videos, labels)
                logits = model(videos)
        
        eval_losses.update(loss.item())
        metrics_tracker.update(logits, labels)
        
        if full_eval:
            epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")
    
    # Compute all metrics
    metrics = metrics_tracker.get_all_metrics()
    
    if full_eval:
        logger.info("\n")
        logger.info(f"Validation Results - Epoch {epoch}")
        logger.info(f"  Loss: {eval_losses.avg:.5f}")
        for name in args.split_names:
            logger.info(f"  {name.capitalize()} Top-1: {metrics[f'{name}_top1']:.5f}")
        logger.info(f"  Joint Accuracy: {metrics['joint_accuracy']:.5f}")
        
        if writer is not None:
            writer.add_scalar("val/loss", eval_losses.avg, global_step)
            for key, value in metrics.items():
                writer.add_scalar(f"val/{key}", value, global_step)
    
    model.train()
    
    return metrics, eval_losses.avg


def train(args, model, config):
    """Main training loop."""
    text_logger = None
    
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        
        log_file_path = os.path.join(args.output_dir, f"{args.name}_training_log.txt")
        text_logger = TextLogger(log_file_path, args, args.class_splits, args.split_names)
        logger.info(f"Training log: {log_file_path}")
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # Get data loaders
    train_loader, test_loader = get_epic_kitchens_loader(args)
    
    if train_loader is None:
        raise ValueError("Training data loader is None. Check your data paths.")
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    t_total = steps_per_epoch * args.num_epochs
    
    # Calculate warmup steps
    if args.warmup_epochs > 0:
        warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    else:
        warmup_steps = args.warmup_steps
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {t_total}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    
    # FP16 Scaler
    scaler = GradScaler() if args.fp16 else None
    
    # Mixup augmentation
    mixup_fn = None
    if args.mixup_alpha > 0:
        mixup_fn = Mixup(alpha=args.mixup_alpha, prob=args.mixup_prob)
        logger.info(f"Using Mixup with alpha={args.mixup_alpha}, prob={args.mixup_prob}")
    
    # EMA (Exponential Moving Average)
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        logger.info(f"Using EMA with decay={args.ema_decay}")
    
    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_metrics = {
        'noun_top1': 0,
        'verb_top1': 0,
        'joint_accuracy': 0
    }
    
    # Resume from checkpoint
    if args.resume is not None:
        start_epoch, global_step, best_metrics = load_checkpoint(
            args, model, optimizer, scheduler, scaler, ema
        )
        if text_logger:
            text_logger.log_message(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Multi-GPU setup
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Training info
    logger.info("***** Running Training *****")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  FP16 = {args.fp16}")
    
    model.zero_grad()
    set_seed(args)
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()
        losses = AverageMeter()
        
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            dynamic_ncols=True,
            disable=args.local_rank not in [-1, 0]
        )
        
        for step, batch in enumerate(epoch_iterator):
            videos, labels = batch
            videos = videos.to(args.device)
            labels = labels.to(args.device)
            
            # Handle video dimension
            if videos.dim() == 4:
                videos = videos.unsqueeze(1).expand(-1, 2, -1, -1, -1)
            
            # Apply Mixup if enabled
            mixup_labels = None
            lam = 1.0
            if mixup_fn is not None:
                videos, labels, mixup_labels, lam = mixup_fn(videos, labels)
            
            # Forward pass
            if args.fp16:
                with autocast():
                    if mixup_labels is not None and lam < 1.0:
                        # Mixup: compute loss as weighted combination
                        loss1 = model(videos, labels)
                        loss2 = model(videos, mixup_labels)
                        loss = lam * loss1 + (1 - lam) * loss2
                    else:
                        loss = model(videos, labels)
            else:
                if mixup_labels is not None and lam < 1.0:
                    loss1 = model(videos, labels)
                    loss2 = model(videos, mixup_labels)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = model(videos, labels)
            
            # Handle DataParallel
            if args.n_gpu > 1 and args.local_rank == -1:
                loss = loss.mean()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                
                # Gradient clipping and optimizer step
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                epoch_iterator.set_description(
                    f"Epoch {epoch}/{args.num_epochs} (loss={losses.val:.5f}, lr={scheduler.get_lr()[0]:.6f})"
                )
                
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                
                # Quick validation
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        metrics, val_loss = validate(
                            args, model, test_loader, epoch, global_step,
                            writer=writer, full_eval=False, scaler=scaler
                        )
                        if text_logger:
                            text_logger.log(global_step, epoch, val_loss, metrics, 
                                          scheduler.get_lr()[0], "quick_eval")
                
                # Save checkpoint
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        save_model(args, model, optimizer, scheduler, scaler, 
                                  epoch, global_step, best_metrics, config,
                                  step_checkpoint=True, ema=ema)
        
        # End of epoch
        logger.info(f"Epoch {epoch} - Average Loss: {losses.avg:.5f}")
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/epoch_loss", losses.avg, epoch)
            
            # Full validation (use EMA weights if available)
            if ema is not None:
                ema.apply_shadow()
            
            metrics, val_loss = validate(
                args, model, test_loader, epoch, global_step,
                writer=writer, full_eval=True, scaler=scaler
            )
            
            if ema is not None:
                ema.restore()
            
            if text_logger:
                text_logger.log(global_step, epoch, val_loss, metrics,
                              scheduler.get_lr()[0], "EPOCH_END")
            
            # Check for best model (based on joint accuracy)
            if metrics['joint_accuracy'] > best_metrics['joint_accuracy']:
                best_metrics = metrics.copy()
                save_model(args, model, optimizer, scheduler, scaler,
                          epoch, global_step, best_metrics, config, best=True, ema=ema)
                logger.info(f"New best joint accuracy: {best_metrics['joint_accuracy']:.5f}")
                if text_logger:
                    text_logger.log_message(f"NEW BEST! Joint: {best_metrics['joint_accuracy']:.5f}")
            
            # Regular checkpoint
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_model(args, model, optimizer, scheduler, scaler,
                          epoch, global_step, best_metrics, config, ema=ema)
    
    if args.local_rank in [-1, 0]:
        writer.close()
        if text_logger:
            text_logger.close(best_metrics)
    
    logger.info("Training completed!")
    logger.info("Best metrics:")
    for key, value in best_metrics.items():
        logger.info(f"  {key}: {value:.5f}")


def main():
    parser = argparse.ArgumentParser(description="ViViT Training for Epic Kitchens")
    
    # Required parameters
    parser.add_argument("--name", default="vivit_epic_kitchens",
                        help="Experiment name")
    parser.add_argument("--model_type", 
                        choices=["ViViT-B/16x2-EK", "ViViT-L/16x2-EK", "ViViT-B/16x2", "ViViT-L/16x2"],
                        default="ViViT-L/16x2-EK",
                        help="Model variant")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-L_16.npz",
                        help="Path to pretrained ViT weights")
    parser.add_argument("--output_dir", default="output_epic_kitchens", type=str,
                        help="Output directory")
    
    # Data parameters
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Path to video files")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test video files (optional)")
    parser.add_argument("--train_annotations", required=True, type=str,
                        help="Path to training annotations CSV")
    parser.add_argument("--val_annotations", type=str, default=None,
                        help="Path to validation annotations CSV")
    parser.add_argument("--test_annotations", type=str, default=None,
                        help="Path to test annotations CSV")
    parser.add_argument("--use_frames", action='store_true',
                        help="Load from pre-extracted frames instead of videos")
    
    # Model parameters
    parser.add_argument("--img_size", default=224, type=int,
                        help="Input image size")
    parser.add_argument("--num_frames", default=32, type=int,
                        help="Number of frames to sample")
    
    # Training parameters
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Training batch size (scenic-vivit default: 64)")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", default=0.5, type=float,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay")
    parser.add_argument("--label_smoothing", default=0.2, type=float,
                        help="Label smoothing factor")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Number of training epochs")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="Learning rate decay type")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Warmup steps")
    parser.add_argument("--warmup_epochs", default=2.5, type=float,
                        help="Warmup epochs (overrides warmup_steps)")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Gradient accumulation steps")
    
    # Checkpoint parameters
    parser.add_argument("--save_every", default=5, type=int,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save_every_steps", default=0, type=int,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_every_steps", default=0, type=int,
                        help="Quick validation every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming")
    
    # Distributed training
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    
    # FP16
    parser.add_argument("--fp16", action='store_true',
                        help="Use FP16 training")
    
    # Regularization
    parser.add_argument("--mixup_alpha", default=0.1, type=float,
                        help="Mixup alpha (0 to disable, scenic-vivit default: 0.1)")
    parser.add_argument("--mixup_prob", default=1.0, type=float,
                        help="Probability of applying mixup")
    parser.add_argument("--use_ema", action='store_true',
                        help="Use Exponential Moving Average of model weights")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="EMA decay rate")
    parser.add_argument("--stochastic_droplayer_rate", default=0.2, type=float,
                        help="Stochastic depth drop rate (scenic-vivit default: 0.2)")
    
    args = parser.parse_args()
    
    # Handle local_rank from environment
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Device setup
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    
    args.device = device
    
    # Logging setup
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    
    logger.info(f"Device: {device}, n_gpu: {args.n_gpu}, distributed: {args.local_rank != -1}")
    
    # Set seed
    set_seed(args)
    
    # Setup model
    args, model, config = setup(args)
    
    # Train
    train(args, model, config)


if __name__ == "__main__":
    main()

