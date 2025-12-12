"""
Regularization utilities for ViViT training.

Includes:
- Mixup data augmentation
- CutMix data augmentation  
- RandAugment
- Stochastic Depth (DropPath)
- Label Smoothing (already in CrossEntropyLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Mixup and CutMix
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation.
    
    Mixes pairs of examples and their labels using beta distribution.
    Reference: https://arxiv.org/abs/1710.09412
    
    Args:
        x: Input tensor of shape (B, ...)
        y: Labels tensor (can be one-hot or class indices)
        alpha: Mixup interpolation strength (beta distribution parameter)
        device: Device to use
        
    Returns:
        mixed_x: Mixed input tensor
        y_a: Original labels
        y_b: Shuffled labels for mixing
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if device is None:
        device = x.device
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute loss with mixup.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Mixup:
    """
    Mixup augmentation class for video data.
    
    Supports both standard Mixup and video-specific Mixup.
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = 'batch',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Mixup interpolation strength
            prob: Probability of applying mixup
            switch_prob: Probability of switching to cutmix (if enabled)
            mode: 'batch' for batch-level mixup, 'elem' for element-wise
            label_smoothing: Label smoothing factor
        """
        self.alpha = alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
    
    def __call__(
        self,
        x: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to batch.
        
        Args:
            x: Input tensor (B, C, T, H, W) or (B, T, C, H, W)
            target: Labels tensor
            
        Returns:
            mixed_x, target_a, target_b, lam
        """
        if np.random.rand() > self.prob:
            return x, target, target, 1.0
        
        return mixup_data(x, target, self.alpha, x.device)


# =============================================================================
# Stochastic Depth (DropPath)
# =============================================================================

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.
    
    Reference: https://arxiv.org/abs/1603.09382
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path
        training: Whether in training mode
        scale_by_keep: Whether to scale output by keep probability
        
    Returns:
        Output tensor with dropped paths
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Work with shape (batch_size, 1, 1, ...) for broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Used in transformer blocks to randomly skip residual connections.
    """
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self) -> str:
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


# =============================================================================
# SAM Optimizer (Sharpness-Aware Minimization)
# =============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer.
    
    SAM seeks parameters that lie in neighborhoods having uniformly low loss,
    resulting in improved generalization.
    
    Reference: https://arxiv.org/abs/2010.01412
    
    Args:
        params: Model parameters
        base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
        rho: Neighborhood size for SAM
        adaptive: Whether to use adaptive SAM
        **kwargs: Arguments for base optimizer
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute perturbation and apply to weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Compute perturbation
                if group["adaptive"]:
                    e_w = (torch.pow(p, 2) * p.grad * scale).to(p)
                else:
                    e_w = (p.grad * scale).to(p)
                
                # Store current weights
                self.state[p]["e_w"] = e_w
                
                # Perturb weights
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore weights and apply gradients."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original weights
                p.sub_(self.state[p]["e_w"])
        
        # Apply base optimizer step
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Single optimization step (for compatibility)."""
        assert closure is not None, "SAM requires closure for gradient computation"
        
        # First forward-backward pass
        closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        self.second_step()
    
    def _grad_norm(self):
        """Compute gradient norm."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# =============================================================================
# RandAugment for Video
# =============================================================================

class RandAugmentVideo:
    """
    RandAugment for video data.
    
    Applies random augmentations consistently across all frames.
    
    Args:
        num_ops: Number of augmentation operations to apply
        magnitude: Magnitude of augmentations (0-10)
        num_magnitude_bins: Number of magnitude bins
    """
    
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31
    ):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        
        # Import torchvision transforms
        try:
            from torchvision.transforms import RandAugment
            self.rand_augment = RandAugment(
                num_ops=num_ops,
                magnitude=magnitude,
                num_magnitude_bins=num_magnitude_bins
            )
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: RandAugment not available. Install torchvision>=0.9.0")
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply RandAugment to video.
        
        Args:
            video: Video tensor of shape (T, C, H, W)
            
        Returns:
            Augmented video tensor
        """
        if not self.available:
            return video
        
        # Apply same augmentation to all frames
        # First, we need to seed the random state for consistency
        seed = np.random.randint(2147483647)
        
        augmented_frames = []
        for t in range(video.shape[0]):
            frame = video[t]
            # Convert to PIL-compatible format if needed
            if frame.dtype == torch.float32:
                frame = (frame * 255).to(torch.uint8)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            augmented = self.rand_augment(frame)
            
            if augmented.dtype == torch.uint8:
                augmented = augmented.float() / 255.0
            
            augmented_frames.append(augmented)
        
        return torch.stack(augmented_frames)


# =============================================================================
# EMA (Exponential Moving Average) Model
# =============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model parameters for improved stability
    and generalization at inference time.
    
    Args:
        model: The model to track
        decay: EMA decay rate (typically 0.999 or 0.9999)
        device: Device to store EMA parameters
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']

