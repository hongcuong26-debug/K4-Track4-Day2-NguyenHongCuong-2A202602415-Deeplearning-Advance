"""Model construction, freezing, optimizer groups, and complexity accounting."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _classifier_param_ids(model) -> set[int]:
    classifier = model.get_classifier()
    modules = classifier if isinstance(classifier, (tuple, list)) else [classifier]
    return {id(parameter) for module in modules if hasattr(module, "parameters")
            for parameter in module.parameters()}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Create a timm classifier under the scratch/frozen/fine-tune protocols."""
    import timm
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be 'scratch', 'frozen', or 'finetune'")
    resolved_name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = bool(pretrained and init != "scratch")
    model = timm.create_model(resolved_name, pretrained=use_pretrained,
                              num_classes=num_classes, drop_rate=drop_rate)
    model.training_recipe = init
    model.pretrained_tag = model.pretrained_cfg.get("tag", "") if use_pretrained else "scratch"
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Freeze all parameters except the classifier and mark the model for BN handling."""
    head_ids = _classifier_param_ids(model)
    if not head_ids:
        raise ValueError("timm model exposes no classifier parameters")
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._frozen_backbone = True


def keep_frozen_backbone_eval(model) -> None:
    """Keep the frozen feature extractor deterministic after ``model.train()``."""
    if not getattr(model, "_frozen_backbone", False):
        return
    model.eval()
    classifier = model.get_classifier()
    modules = classifier if isinstance(classifier, (tuple, list)) else [classifier]
    for module in modules:
        if hasattr(module, "train"):
            module.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Return backbone-decay, backbone-no-decay, and classifier parameter groups."""
    if min(lr_backbone, lr_head, weight_decay) < 0:
        raise ValueError("learning rates and weight decay must be non-negative")
    head_ids = _classifier_param_ids(model)
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups["head"].append(parameter)
        elif parameter.ndim <= 1 or name.endswith(".bias"):
            groups["no_decay"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    result = []
    specs = (("backbone", lr_backbone, weight_decay), ("no_decay", lr_backbone, 0.0),
             ("head", lr_head, weight_decay))
    for key, lr, decay in specs:
        if groups[key]:
            result.append({"params": groups[key], "lr": lr, "weight_decay": decay,
                           "group_name": key})
    if not result:
        raise ValueError("model has no trainable parameters")
    return result


def count_params(model) -> float:
    """Count all model parameters in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """Count Conv/Linear MACs with hooks for one image (attention ops are not included)."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    macs = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups
        macs += int(out.numel() * kernel_ops)

    def linear_hook(module, inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        macs += int(out.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=parameter.device,
                              dtype=parameter.dtype))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return macs / 1e9
