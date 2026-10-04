"""Deterministic inference, TTA, ensembles, calibration, and Conv-BN fusion."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    """Collect filenames, labels, and logits in stable loader order."""
    model.eval()
    names, labels, outputs = [], [], []
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            logits = model(images)
            names.extend(list(filenames))
            labels.append(target.detach().cpu())
            outputs.append(logits.float().detach().cpu())
    return names, torch.cat(labels).numpy(), torch.cat(outputs).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return four corner crops and a center crop."""
    if x.ndim != 4:
        raise ValueError("x must be NCHW")
    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError("crop must fit inside the image")
    top, left = height - crop, width - crop
    center_y, center_x = top // 2, left // 2
    locations = ((0, 0), (0, left), (top, 0), (top, left), (center_y, center_x))
    return [x[:, :, y:y + crop, z:z + crop] for y, z in locations]


def views_multiscale(x, sizes):
    if x.ndim != 4:
        raise ValueError("x must be NCHW")
    result = []
    for size in sizes:
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("each size must be a positive int or (height, width)")
        result.append(F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False,
                                    antialias=True))
    return result


def _stack(values):
    any_numpy = all(isinstance(value, np.ndarray) for value in values)
    tensors = [torch.as_tensor(value) for value in values]
    return torch.stack(tensors), any_numpy


def aggregate_views(logits_per_view, space: str = "prob"):
    """Average view predictions in probability or logit space."""
    if not logits_per_view:
        raise ValueError("at least one view is required")
    stacked, return_numpy = _stack(logits_per_view)
    if space == "prob":
        probs = stacked.softmax(dim=-1).mean(dim=0)
    elif space == "logit":
        probs = stacked.mean(dim=0).softmax(dim=-1)
    else:
        raise ValueError("space must be 'prob' or 'logit'")
    return probs.numpy() if return_numpy else probs


def ensemble_probs(list_of_probs):
    """Average aligned probability matrices and re-normalize defensively."""
    if not list_of_probs:
        raise ValueError("at least one model is required")
    stacked, return_numpy = _stack(list_of_probs)
    if stacked.ndim != 3 or torch.any(stacked < 0):
        raise ValueError("probabilities must have shape (models, samples, classes) and be non-negative")
    probs = stacked.mean(dim=0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
    return probs.numpy() if return_numpy else probs


def fit_temperature(val_logits, val_labels) -> float:
    """Fit one positive temperature by minimizing validation NLL."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("expected logits[N,K] and labels[N]")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 100.0))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    is_numpy = isinstance(logits, np.ndarray)
    probs = torch.as_tensor(logits).div(float(T)).softmax(dim=-1)
    return probs.numpy() if is_numpy else probs


def _fuse_children(module):
    names = list(module._modules)
    index = 0
    while index + 1 < len(names):
        first, second = names[index], names[index + 1]
        conv, bn = module._modules[first], module._modules[second]
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            module._modules[first] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            module._modules[second] = nn.Identity()
            index += 2
        else:
            _fuse_children(conv)
            index += 1
    if names:
        _fuse_children(module._modules[names[-1]])


def fuse_conv_bn(model):
    """Return an eval-mode deep copy with adjacent Conv2d/BatchNorm2d pairs fused."""
    fused = copy.deepcopy(model).eval()
    _fuse_children(fused)
    return fused
