# coding=utf-8
"""
ViViT Training Script with Native PyTorch AMP (No Apex Required!)
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler  # Native PyTorch AMP - No Apex needed!

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import CONFIGS, MyViViT

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """
    Computes top-k accuracy.
    """
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(labels)


def save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, best_top1_acc, best_top5_acc, best=False, step_checkpoint=False):
    """
    Save model checkpoint with all training state for resume capability.
    """
    config = CONFIGS[args.model_type]
    config.num_classes = args.num_classes
    config.img_size = args.img_size
    config.num_frames = args.num_frames
    
    model_to_save = model.module if hasattr(model, 'module') else model
   # NEW:
    if best:
        model_checkpoint = os.path.join(args.output_dir, "%s_best.pth" % args.name)
    elif step_checkpoint:
        model_checkpoint = os.path.join(args.output_dir, "%s_step_%d.pth" % (args.name, global_step))
    else:
        model_checkpoint = os.path.join(args.output_dir, "%s_epoch_%d.pth" % (args.name, epoch))
    
    checkpoint = {
        'config': config,
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_top1_acc': best_top1_acc,
        'best_top5_acc': best_top5_acc,
        'args': args,
    }
    
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [%s]", model_checkpoint)


def load_checkpoint(args, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load checkpoint for resuming training.
    """
    logger.info(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Loaded scheduler state")
    
    if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info("Loaded scaler state")
    
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_top1_acc = checkpoint.get('best_top1_acc', 0)
    best_top5_acc = checkpoint.get('best_top5_acc', 0)
    
    logger.info(f"Resumed from epoch {start_epoch}, global_step {global_step}")
    logger.info(f"Best Top-1 Acc: {best_top1_acc:.5f}, Best Top-5 Acc: {best_top5_acc:.5f}")
    
    return start_epoch, global_step, best_top1_acc, best_top5_acc


def setup(args):
    config = CONFIGS[args.model_type]
    config.label_smoothing = args.label_smoothing

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        num_frames = 2
        num_classes = 10 if args.dataset == "cifar10" else 100
    else:
        num_frames = args.num_frames
        num_classes = args.num_classes
        
    model = MyViViT(config, args.img_size, num_classes=num_classes, num_frames=num_frames)
    
    if args.resume is None:
        model.load_from(np.load(args.pretrained_dir))
    
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, epoch, global_step, full_eval=True, scaler=None):
    """
    Run validation with optional FP16.
    """
    eval_losses = AverageMeter()

    if full_eval:
        logger.info("***** Running Validation *****")
        logger.info("  Num steps = %d", len(test_loader))
        logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label, all_logits = [], [], []
    
    max_batches = len(test_loader) if full_eval else min(10, len(test_loader))
    
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0] or not full_eval)
    
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        if step >= max_batches:
            break
            
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        
        if x.dim() == 4:
            x = torch.unsqueeze(x, 1)
            x = x.expand(-1, 2, -1, -1, -1)
            
        with torch.no_grad():
            # Use autocast for validation too if fp16 is enabled
            if args.fp16:
                with autocast():
                    logits = model(x)
                    eval_loss = loss_fct(logits, y)
            else:
                logits = model(x)
                eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(logits.float().detach().cpu().numpy())  # Convert to float for numpy
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
            all_logits[0] = np.append(all_logits[0], logits.float().detach().cpu().numpy(), axis=0)
            
        if full_eval:
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_logits = all_preds[0], all_label[0], all_logits[0]
    
    top1_accuracy = simple_accuracy(all_preds, all_label)
    num_classes = all_logits.shape[1]
    k = min(5, num_classes)
    top5_accuracy = top_k_accuracy(all_logits, all_label, k=k)

    if full_eval:
        logger.info("\n")
        logger.info("Validation Results - Epoch %d" % epoch)
        logger.info("Valid Loss: %2.5f" % eval_losses.avg)
        logger.info("Valid Top-1 Accuracy: %2.5f" % top1_accuracy)
        logger.info("Valid Top-5 Accuracy: %2.5f" % top5_accuracy)

        writer.add_scalar("test/top1_accuracy", scalar_value=top1_accuracy, global_step=global_step)
        writer.add_scalar("test/top5_accuracy", scalar_value=top5_accuracy, global_step=global_step)
        writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=global_step)
    else:
        logger.info(f"[Step {global_step}] Quick Eval - Loss: {eval_losses.avg:.5f}, Top-1 Acc: {top1_accuracy:.5f}, Top-5 Acc: {top5_accuracy:.5f}")
        writer.add_scalar("test/quick_top1_accuracy", scalar_value=top1_accuracy, global_step=global_step)
        writer.add_scalar("test/quick_top5_accuracy", scalar_value=top5_accuracy, global_step=global_step)
    
    model.train()
    
    return top1_accuracy, top5_accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Calculate total steps based on epochs
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    t_total = steps_per_epoch * args.num_epochs

    # Calculate warmup steps
    if args.warmup_epochs > 0:
        warmup_steps = int(steps_per_epoch * args.warmup_epochs)
        logger.info(f"Using warmup_epochs={args.warmup_epochs}, calculated warmup_steps={warmup_steps}")
    else:
        warmup_steps = args.warmup_steps
        logger.info(f"Using warmup_steps={warmup_steps}")

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # Initialize GradScaler for FP16 training (Native PyTorch AMP)
    scaler = GradScaler() if args.fp16 else None
    if args.fp16:
        logger.info("Using Native PyTorch AMP (FP16 training enabled)")

    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_top1_acc = 0
    best_top5_acc = 0
    resume_step_in_epoch = 0

    # Resume from checkpoint if specified
    if args.resume is not None:
        start_epoch, global_step, best_top1_acc, best_top5_acc = load_checkpoint(
            args, model, optimizer, scheduler, scaler
        )
        resume_step_in_epoch = global_step % steps_per_epoch
        if resume_step_in_epoch > 0:
            logger.info(f"Will skip first {resume_step_in_epoch} steps in epoch {start_epoch + 1}")

    # Multi-GPU setup
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        logger.info(f"Using DistributedDataParallel on GPU {args.local_rank}")
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using DataParallel on {args.n_gpu} GPUs")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num epochs = %d", args.num_epochs)
    logger.info("  Start epoch = %d", start_epoch + 1)
    logger.info("  Steps per epoch = %d", steps_per_epoch)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", warmup_steps)
    logger.info("  FP16 training = %s", args.fp16)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Number of GPUs = %d", args.n_gpu if args.local_rank == -1 else torch.distributed.get_world_size())
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else max(1, args.n_gpu)))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()
        losses = AverageMeter()
        
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        epoch_iterator = tqdm(train_loader,
                              desc=f"Epoch {epoch}/{args.num_epochs} (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            if epoch == start_epoch + 1 and step < resume_step_in_epoch:
                continue
            
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            
            if x.dim() == 4:
                x = torch.unsqueeze(x, 1)
                x = x.expand(-1, 2, -1, -1, -1)
            
            # Forward pass with autocast for FP16
            if args.fp16:
                with autocast():
                    loss = model(x, y)
            else:
                loss = model(x, y)
            
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

                epoch_iterator.set_description(
                    f"Epoch {epoch}/{args.num_epochs} (loss={losses.val:.5f}, lr={scheduler.get_lr()[0]:.6f})"
                )
                
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
                # Quick evaluation
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        valid(args, model, writer, test_loader, epoch, global_step, full_eval=False, scaler=scaler)
                
                # Save checkpoint
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    if args.local_rank in [-1, 0]:
                        save_model(args, model, optimizer, scheduler, scaler, epoch, global_step,
                                  best_top1_acc, best_top5_acc, best=False, step_checkpoint=True)

        resume_step_in_epoch = 0
        logger.info(f"Epoch {epoch}/{args.num_epochs} completed - Average Loss: {losses.avg:.5f}")
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/epoch_loss", scalar_value=losses.avg, global_step=epoch)
            
            top1_accuracy, top5_accuracy = valid(args, model, writer, test_loader, epoch, global_step, full_eval=True, scaler=scaler)
            
            if best_top1_acc < top1_accuracy:
                save_model(args, model, optimizer, scheduler, scaler, epoch, global_step, 
                          top1_accuracy, top5_accuracy, best=True)
                best_top1_acc = top1_accuracy
                best_top5_acc = top5_accuracy
                logger.info(f"New best Top-1 accuracy: {best_top1_acc:.5f}, Top-5 accuracy: {best_top5_acc:.5f}")
            
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_model(args, model, optimizer, scheduler, scaler, epoch, global_step,
                          best_top1_acc, best_top5_acc, best=False)

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Top-1 Accuracy: \t%f" % best_top1_acc)
    logger.info("Best Top-5 Accuracy: \t%f" % best_top5_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False, default="test",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "custom"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViViT-B/16x2", "ViViT-B/16x2-small", "ViViT-L/16x2"],
                        default="ViViT-L/16x2",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--data_dir", default="train", type=str,
                        help="Path to the data.")
    parser.add_argument("--test_dir", default=None, type=str,
                        help="Path to the test data.")
    parser.add_argument("--num_classes", default=174, type=int,
                        help="number of classes")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--num_frames", default=16, type=int,
                        help="Number of input frames to sample")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--label_smoothing", default=0.3, type=float,
                        help="label smoothing p.")
    parser.add_argument("--num_epochs", default=35, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--warmup_epochs", default=2.5, type=float,
                        help="Number of warmup epochs (overrides warmup_steps if > 0).")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--save_every", default=1, type=int,
                        help="Save checkpoint every N epochs (0 = only save best).")
    parser.add_argument("--save_every_steps", default=50, type=int,
                        help="Save checkpoint every N steps (0 = disabled).")
    parser.add_argument("--eval_every_steps", default=50, type=int,
                        help="Run quick validation every N steps (0 = only at epoch end).")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward/update pass.")
    
    # FP16 - Uses Native PyTorch AMP (no Apex needed!)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help="Whether to use 16-bit float precision (Native PyTorch AMP)")

    args = parser.parse_args()

    # Handle local_rank from environment variable (for torchrun)
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()