# coding=utf-8
"""
ViViT Training Script with Epic Kitchens Support and Full Regularization

Supports:
- Standard classification (SSv2, Kinetics, etc.)
- Epic Kitchens multi-head classification (verb + noun -> action)
- Mixup augmentation
- Label smoothing
- All ViViT paper regularization (Table 7)
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import CONFIGS, MyViViT
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, DATASET_CONFIGS
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


# =============================================================================
# Epic Kitchens Multi-Head Functions
# =============================================================================

def epic_kitchens_loss(logits, labels, class_splits, label_smoothing=0.0):
    """
    Compute loss for Epic Kitchens multi-head classification.
    
    Args:
        logits: [B, 397] model outputs
        labels: [B, 397] one-hot labels (noun | verb concatenated)
        class_splits: cumulative splits [300, 397]
        label_smoothing: smoothing factor
    
    Returns:
        Average of noun and verb cross-entropy losses
    """
    # Split logits and labels
    noun_logits = logits[:, :class_splits[0]]  # [B, 300]
    verb_logits = logits[:, class_splits[0]:]  # [B, 97]
    
    noun_labels = labels[:, :class_splits[0]]
    verb_labels = labels[:, class_splits[0]:]
    
    # Check if soft labels (from mixup)
    is_soft = noun_labels.sum(dim=-1).mean() > 1.01
    
    if is_soft:
        # Soft cross-entropy for mixup
        noun_log_probs = F.log_softmax(noun_logits, dim=-1)
        verb_log_probs = F.log_softmax(verb_logits, dim=-1)
        
        noun_loss = -(noun_labels * noun_log_probs).sum(dim=-1).mean()
        verb_loss = -(verb_labels * verb_log_probs).sum(dim=-1).mean()
    else:
        # Standard cross-entropy
        noun_targets = noun_labels.argmax(dim=-1)
        verb_targets = verb_labels.argmax(dim=-1)
        
        noun_loss = F.cross_entropy(noun_logits, noun_targets, label_smoothing=label_smoothing)
        verb_loss = F.cross_entropy(verb_logits, verb_targets, label_smoothing=label_smoothing)
    
    return (noun_loss + verb_loss) / 2


def epic_kitchens_accuracy(logits, labels, class_splits):
    """
    Compute accuracies for Epic Kitchens.
    
    Returns:
        action_acc: Both noun AND verb correct (primary metric)
        noun_acc: Noun Top-1 accuracy
        verb_acc: Verb Top-1 accuracy
    """
    noun_logits = logits[:, :class_splits[0]]
    verb_logits = logits[:, class_splits[0]:]
    
    noun_labels = labels[:, :class_splits[0]]
    verb_labels = labels[:, class_splits[0]:]
    
    noun_pred = noun_logits.argmax(dim=-1)
    verb_pred = verb_logits.argmax(dim=-1)
    
    noun_true = noun_labels.argmax(dim=-1)
    verb_true = verb_labels.argmax(dim=-1)
    
    noun_correct = (noun_pred == noun_true)
    verb_correct = (verb_pred == verb_true)
    action_correct = noun_correct & verb_correct
    
    return (
        action_correct.float().mean().item(),
        noun_correct.float().mean().item(),
        verb_correct.float().mean().item()
    )


# =============================================================================
# Mixup Implementation
# =============================================================================

def mixup_data(x, y, alpha, is_onehot=False):
    """
    Apply mixup augmentation.
    
    Args:
        x: Input tensor [B, ...]
        y: Labels - [B] for class indices, [B, C] for one-hot
        alpha: Beta distribution parameter (0 = no mixup)
        is_onehot: Whether y is already one-hot encoded
    
    Returns:
        mixed_x, mixed_y, lambda
    """
    if alpha <= 0:
        return x, y, 1.0
    
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size, device=x.device)
    
    # Mix inputs
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Mix labels
    if is_onehot:
        mixed_y = lam * y + (1 - lam) * y[index]
    else:
        # Return tuple for standard classification
        mixed_y = (y, y[index], lam)
    
    return mixed_x, mixed_y, lam


def mixup_criterion(logits, y_tuple, label_smoothing=0.0):
    """Compute mixup loss for standard classification."""
    y_a, y_b, lam = y_tuple
    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1 - lam) * loss_b


# =============================================================================
# Utility Classes
# =============================================================================

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def top_k_accuracy(logits, labels, k=5):
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i])
    return correct / len(labels)


def save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, 
               best_acc, best=False, step_checkpoint=False):
    """Save model checkpoint."""
    model_to_save = model.module if hasattr(model, 'module') else model
    
    if best:
        path = os.path.join(args.output_dir, f"{args.name}_best.pth")
    elif step_checkpoint:
        path = os.path.join(args.output_dir, f"{args.name}_step_{global_step}.pth")
    else:
        path = os.path.join(args.output_dir, f"{args.name}_epoch_{epoch}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_acc': best_acc,
        'args': args,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(args, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint for resuming."""
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('global_step', 0),
        checkpoint.get('best_acc', 0)
    )


def setup(args):
    """Setup model."""
    config = CONFIGS[args.model_type]
    config.label_smoothing = args.label_smoothing
    
    if args.dataset in ["cifar10", "cifar100"]:
        num_frames = 2
        num_classes = 10 if args.dataset == "cifar10" else 100
    else:
        num_frames = args.num_frames
        num_classes = args.num_classes
        
    model = MyViViT(config, args.img_size, num_classes=num_classes, num_frames=num_frames)
    
    if args.resume is None:
        model.load_from(np.load(args.pretrained_dir))
    
    model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Model: {args.model_type}, Parameters: {num_params:.1f}M")
    
    return model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# =============================================================================
# Validation
# =============================================================================

def validate(args, model, test_loader, epoch, global_step, writer, full_eval=True):
    """Run validation."""
    model.eval()
    
    is_epic = args.dataset == 'epic_kitchens' and args.class_splits is not None
    if is_epic:
        class_splits = np.cumsum(args.class_splits).tolist()
    
    losses = AverageMeter()
    
    if is_epic:
        action_acc_meter = AverageMeter()
        noun_acc_meter = AverageMeter()
        verb_acc_meter = AverageMeter()
    else:
        all_preds, all_labels, all_logits = [], [], []
    
    max_batches = len(test_loader) if full_eval else min(10, len(test_loader))
    
    iterator = tqdm(test_loader, desc="Validating", disable=args.local_rank not in [-1, 0] or not full_eval)
    
    for step, batch in enumerate(iterator):
        if step >= max_batches:
            break
        
        x, y = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            if args.fp16:
                with autocast():
                    logits = model(x)
            else:
                logits = model(x)
            
            if is_epic:
                loss = epic_kitchens_loss(logits, y, class_splits)
                action_acc, noun_acc, verb_acc = epic_kitchens_accuracy(logits, y, class_splits)
                
                losses.update(loss.item(), x.size(0))
                action_acc_meter.update(action_acc, x.size(0))
                noun_acc_meter.update(noun_acc, x.size(0))
                verb_acc_meter.update(verb_acc, x.size(0))
            else:
                loss = F.cross_entropy(logits, y)
                losses.update(loss.item(), x.size(0))
                
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_logits.extend(logits.float().cpu().numpy())
    
    # Compute final metrics
    if is_epic:
        if full_eval and args.local_rank in [-1, 0]:
            logger.info(f"\nValidation Results - Epoch {epoch}")
            logger.info(f"Loss: {losses.avg:.5f}")
            logger.info(f"Action Accuracy: {action_acc_meter.avg:.5f}")
            logger.info(f"Noun Accuracy: {noun_acc_meter.avg:.5f}")
            logger.info(f"Verb Accuracy: {verb_acc_meter.avg:.5f}")
            
            writer.add_scalar("val/loss", losses.avg, global_step)
            writer.add_scalar("val/action_accuracy", action_acc_meter.avg, global_step)
            writer.add_scalar("val/noun_accuracy", noun_acc_meter.avg, global_step)
            writer.add_scalar("val/verb_accuracy", verb_acc_meter.avg, global_step)
        
        model.train()
        return action_acc_meter.avg
    else:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        top1_acc = simple_accuracy(all_preds, all_labels)
        top5_acc = top_k_accuracy(all_logits, all_labels, k=min(5, all_logits.shape[1]))
        
        if full_eval and args.local_rank in [-1, 0]:
            logger.info(f"\nValidation Results - Epoch {epoch}")
            logger.info(f"Loss: {losses.avg:.5f}")
            logger.info(f"Top-1 Accuracy: {top1_acc:.5f}")
            logger.info(f"Top-5 Accuracy: {top5_acc:.5f}")
            
            writer.add_scalar("val/loss", losses.avg, global_step)
            writer.add_scalar("val/top1_accuracy", top1_acc, global_step)
            writer.add_scalar("val/top5_accuracy", top5_acc, global_step)
        
        model.train()
        return top1_acc


# =============================================================================
# Training
# =============================================================================

def train(args, model):
    """Main training loop."""
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    
    # Adjust batch size for gradient accumulation
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # Dataset setup
    is_epic = args.dataset == 'epic_kitchens' and args.class_splits is not None
    if is_epic:
        class_splits = np.cumsum(args.class_splits).tolist()
        logger.info(f"Epic Kitchens: {args.class_splits[0]} nouns + {args.class_splits[1]} verbs")
    
    # Get mixup alpha
    mixup_alpha = getattr(args, 'mixup_alpha', 0.0)
    
    # Load data
    train_loader, test_loader = get_loader(args)
    
    # Calculate steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    t_total = steps_per_epoch * args.num_epochs
    
    warmup_steps = int(steps_per_epoch * args.warmup_epochs) if args.warmup_epochs > 0 else args.warmup_steps
    
    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total) \
        if args.decay_type == "cosine" else \
        WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    
    # FP16 scaler
    scaler = GradScaler() if args.fp16 else None
    
    # Resume
    start_epoch, global_step, best_acc = 0, 0, 0
    if args.resume:
        start_epoch, global_step, best_acc = load_checkpoint(args, model, optimizer, scheduler, scaler)
    
    # Distributed
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Training info
    logger.info("***** Training *****")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Steps/epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {t_total}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Mixup alpha: {mixup_alpha}")
    
    model.zero_grad()
    set_seed(args)
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()
        losses = AverageMeter()
        
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                       disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(iterator):
            x, y = tuple(t.to(args.device) for t in batch)
            
            # Apply mixup
            if mixup_alpha > 0:
                x, y, lam = mixup_data(x, y, mixup_alpha, is_onehot=is_epic)
            
            # Forward
            if args.fp16:
                with autocast():
                    logits = model(x)
                    
                    if is_epic:
                        loss = epic_kitchens_loss(logits, y, class_splits, args.label_smoothing)
                    elif mixup_alpha > 0:
                        loss = mixup_criterion(logits, y, args.label_smoothing)
                    else:
                        loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            else:
                logits = model(x)
                
                if is_epic:
                    loss = epic_kitchens_loss(logits, y, class_splits, args.label_smoothing)
                elif mixup_alpha > 0:
                    loss = mixup_criterion(logits, y, args.label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            
            if args.n_gpu > 1 and args.local_rank == -1:
                loss = loss.mean()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # Backward
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                
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
                
                iterator.set_description(
                    f"Epoch {epoch}/{args.num_epochs} (loss={losses.val:.4f}, lr={scheduler.get_lr()[0]:.6f})"
                )
                
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                
                # Quick eval
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        validate(args, model, test_loader, epoch, global_step, writer, full_eval=False)
                
                # Step checkpoint
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        save_model(args, model, optimizer, scheduler, scaler, epoch, global_step,
                                  best_acc, step_checkpoint=True)
        
        # Epoch end
        logger.info(f"Epoch {epoch} - Loss: {losses.avg:.5f}")
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/epoch_loss", losses.avg, epoch)
            
            # Full validation
            val_acc = validate(args, model, test_loader, epoch, global_step, writer, full_eval=True)
            
            # Save best
            if val_acc > best_acc:
                save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, val_acc, best=True)
                best_acc = val_acc
                logger.info(f"New best accuracy: {best_acc:.5f}")
            
            # Epoch checkpoint
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, best_acc)
    
    if args.local_rank in [-1, 0]:
        writer.close()
    
    logger.info(f"Best Accuracy: {best_acc:.5f}")
    logger.info("Training complete!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Basic
    parser.add_argument("--name", default="vivit", help="Run name")
    parser.add_argument("--dataset", default="custom",
                        choices=["cifar10", "cifar100", "custom", "epic_kitchens", "ssv2",
                                "kinetics400", "kinetics600", "moments_in_time"])
    parser.add_argument("--model_type", default="ViViT-B/16x2",
                        choices=["ViViT-B/16x2", "ViViT-B/16x2-small", "ViViT-L/16x2"])
    parser.add_argument("--pretrained_dir", default="ViT-B_16.npz")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--data_dir", default="train")
    parser.add_argument("--test_dir", default=None)
    
    # Model
    parser.add_argument("--num_classes", type=int, default=174)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--class_splits", type=int, nargs='+', default=None,
                        help="For Epic Kitchens: 300 97")
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--decay_type", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=float, default=2.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Regularization
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    
    # Distributed
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    
    # Augmentation flag
    parser.add_argument("--use_vivit_aug", action="store_true",
                        help="Use ViViT paper augmentations")
    
    args = parser.parse_args()
    
    # Handle torchrun
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Apply dataset defaults
    if args.dataset in DATASET_CONFIGS:
        config = DATASET_CONFIGS[args.dataset]
        if args.class_splits is None and 'class_splits' in config:
            args.class_splits = config['class_splits']
        if args.num_classes == 174:
            args.num_classes = config.get('num_classes', 174)
    
    # Epic Kitchens validation
    if args.dataset == 'epic_kitchens':
        if args.class_splits is None:
            args.class_splits = [300, 97]
        args.num_classes = sum(args.class_splits)
    
    # CUDA setup
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    
    logger.info(f"Device: {args.device}, GPUs: {args.n_gpu}, FP16: {args.fp16}")
    
    set_seed(args)
    model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()