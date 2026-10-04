"""Classification losses plus batch-level Mixup and CutMix."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    """Build CE, label-smoothed CE, focal, or class-weighted CE."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted requires weight=tensor")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"unknown loss kind: {kind!r}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy using q(k)=(1-eps)1[k=y]+eps/K."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma=0 is exactly weighted/unweighted CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if tensor.ndim != 1:
                raise ValueError("alpha must be a one-dimensional class-weight vector")
            self.register_buffer("alpha", tensor)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Compute normalized inverse-frequency or effective-number weights."""
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or len(values) != 9 or torch.any(values <= 0):
        raise ValueError("counts must contain nine positive class counts")
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1)")
    if beta == 0:
        weights = values.reciprocal()
    else:
        # expm1/log formulation is stable when beta is close to one.
        weights = (1.0 - beta) / (-torch.expm1(values * math.log(beta)))
    weights = weights * (len(values) / weights.sum())
    return weights.float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Apply Mixup or CutMix and return mixed images plus paired hard labels."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be 'mixup' or 'cutmix'")
    if x.ndim != 4 or len(x) != len(y):
        raise ValueError("x must be NCHW and match y's batch dimension")
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    permutation = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        mixed = x * lam + x[permutation] * (1.0 - lam)
    else:
        height, width = x.shape[-2:]
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
        center_x = int(torch.randint(width, (1,), device=x.device))
        center_y = int(torch.randint(height, (1,), device=x.device))
        x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
        y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y, y[permutation], lam)


def mixed_loss(criterion, logits, targets):
    """Evaluate a hard-label criterion on Mixup/CutMix paired targets."""
    y_a, y_b, lam = targets
    return float(lam) * criterion(logits, y_a) + (1.0 - float(lam)) * criterion(logits, y_b)
