"""Reproducible experiment registry for the backbone and one-factor ablation stages."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from train import Config, run, run_dir

BACKBONES = {
    "B01": "resnet50",
    "B02": "resnext50",
    "B03": "convnext_tiny",
    "B04": "deit_small",
    "B05": "swin_tiny",
    "B06": "efficientnet_b0",
}

# Every entry differs from T00 in exactly one controlled factor.
ABLATIONS = {
    "T00": {},
    "T01": {"init": "scratch"},
    "T02": {"init": "frozen"},
    "T03": {"aug": "color"},
    "T04": {"aug": "randaug"},
    "T05": {"loss": "ls", "label_smoothing": 0.1},
    "T06": {"loss": "focal", "focal_gamma": 2.0},
    "T07": {"loss": "ce_weighted", "class_weight_beta": 0.9999},
    "T08": {"mix": "mixup", "mix_alpha": 0.4},
    "T09": {"mix": "cutmix", "mix_alpha": 1.0},
    "T10": {"ema_decay": 0.999},
}


def registry(stage: str, backbone: str, seed: int):
    base = Config(seed=seed, backbone=backbone)
    if stage == "backbones":
        return [replace(base, exp_id=exp_id, backbone=name) for exp_id, name in BACKBONES.items()]
    return [replace(base, exp_id=exp_id, **changes) for exp_id, changes in ABLATIONS.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("backbones", "ablations"), required=True)
    parser.add_argument("--backbone", default="convnext_tiny",
                        help="Backbone used for ablations after validation-based selection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = registry(args.stage, args.backbone, args.seed)
    if args.dry_run:
        print(json.dumps([cfg.__dict__ for cfg in configs], indent=2))
        return
    for cfg in configs:
        summary = run_dir(cfg) / "summary.json"
        if summary.exists() and not args.force:
            print(f"SKIP {cfg.exp_id}: {summary} exists")
            continue
        print(f"RUN {cfg.exp_id}: {cfg.backbone}")
        print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
