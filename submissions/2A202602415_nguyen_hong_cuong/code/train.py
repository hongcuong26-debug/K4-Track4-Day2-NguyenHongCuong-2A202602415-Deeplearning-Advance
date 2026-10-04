"""Single configurable training pipeline for all DeepWeeds experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as data_module
import losses
import model as model_module

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "submissions/2A202602415_nguyen_hong_cuong/predictions"
    curves_dir: str = "submissions/2A202602415_nguyen_hong_cuong/curves"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    groups = model_module.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Per-iteration linear warmup followed by cosine decay."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = max(0, int(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Exponential moving average of parameters, with exact buffer copies."""

    def __init__(self, model, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                target.copy_(value)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    model_module.keep_frozen_backbone_eval(model)
    running_loss, examples = 0.0, 0
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    started = time.perf_counter()
    for images, target, _ in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed_target = None
        if cfg.mix:
            images, mixed_target = losses.mix_batch(images, target, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = (losses.mixed_loss(criterion, logits, mixed_target)
                    if mixed_target is not None else criterion(logits, target))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        batch = len(target)
        running_loss += float(loss.detach()) * batch
        examples += batch
    return {"train_loss": running_loss / max(1, examples),
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started}


def evaluate(model, loader, criterion, device):
    model.eval()
    names, labels, all_logits = [], [], []
    total_loss, examples = 0.0, 0
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, target)
            batch = len(target)
            total_loss += float(loss) * batch
            examples += batch
            names.extend(list(filenames))
            labels.append(target.cpu())
            all_logits.append(logits.float().cpu())
    return names, torch.cat(labels).numpy(), torch.cat(all_logits).numpy(), total_loss / max(1, examples)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, left = plt.subplots(figsize=(8, 5))
    left.plot(frame["epoch"], frame["train_loss"], label="train loss")
    left.plot(frame["epoch"], frame["val_loss"], label="val loss")
    left.set(xlabel="epoch", ylabel="loss")
    right = left.twinx()
    right.plot(frame["epoch"], frame["val_macro_f1"], color="tab:green", label="val macro-F1")
    right.set_ylabel("macro-F1")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    left.grid(alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _criterion(cfg: Config, train_df: pd.DataFrame, device):
    kwargs = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        beta = 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta
        counts = train_df["Label"].value_counts().reindex(range(9), fill_value=0).to_numpy()
        kwargs["weight"] = losses.class_weights(counts, beta).to(device)
    return losses.build_criterion(cfg.loss, **kwargs)


def _save_split(model, frame, cfg, split, criterion, device):
    transform = data_module.build_transforms(False, cfg.img_size, cfg.aug)
    loader = data_module.make_loader(frame, cfg.images_dir, transform, cfg.batch_size,
                                     False, None, cfg.num_workers)
    names, labels, logits, loss = evaluate(model, loader, criterion, device)
    probs = torch.from_numpy(logits).softmax(dim=1).numpy()
    save_predictions(pred_path(cfg, split), names, labels, probs)
    np.savez_compressed(run_dir(cfg) / f"{split}_logits.npz", filenames=np.asarray(names),
                        y_true=labels, logits=logits)
    return compute_metrics(labels, probs.argmax(1), probs), loss


def run(cfg: Config) -> dict:
    """Train one configuration, selecting checkpoints only by validation macro-F1."""
    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_df, val_df, test_df = data_module.load_split(cfg.labels_dir, cfg.fold)
    split_stats = data_module.check_split(train_df, val_df, test_df, cfg.images_dir)
    (output / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    train_loader = data_module.make_loader(
        train_df, cfg.images_dir, data_module.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = data_module.make_loader(
        val_df, cfg.images_dir, data_module.build_transforms(False, cfg.img_size, cfg.aug),
        cfg.batch_size, False, None, cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_module.build_model(cfg.backbone, init=cfg.init,
                                     drop_rate=cfg.drop_rate).to(device)
    criterion = _criterion(cfg, train_df, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    checkpoint = output / "best.pt"
    history, best_f1, best_epoch = [], -1.0, -1

    for epoch in range(cfg.epochs):
        row = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                              cfg, device, ema)
        evaluation_model = ema.module if ema is not None else model
        _, labels, logits, val_loss = evaluate(evaluation_model, val_loader, criterion, device)
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        metrics = compute_metrics(labels, probs.argmax(1), probs)
        row.update({"epoch": epoch + 1, "val_loss": val_loss,
                    "val_top1": metrics["top1"], "val_macro_f1": metrics["macro_f1"],
                    "val_balanced_acc": metrics["balanced_acc"]})
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = float(metrics["macro_f1"]), epoch + 1
            torch.save({"model": evaluation_model.state_dict(), "epoch": best_epoch,
                        "val_macro_f1": best_f1, "config": asdict(cfg)}, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    val_metrics, _ = _save_split(model, val_df, cfg, "val", criterion, device)
    test_metrics = None
    if cfg.save_test_predictions:
        test_metrics, _ = _save_split(model, test_df, cfg, "test", criterion, device)
    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} {cfg.backbone} seed {cfg.seed}")
    summary = {
        "best_epoch": best_epoch, "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1"], "params_m": model_module.count_params(model),
        "gmacs_conv_linear": model_module.count_gmacs(model, cfg.img_size),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "test_macro_f1": None if test_metrics is None else test_metrics["macro_f1"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    valid = {field.name for field in fields(Config)}
    defaults = asdict(Config())
    optional_float = {"class_weight_beta", "ema_decay"}
    optional_string = {"sampler", "mix"}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be KEY=VALUE: {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in valid:
            raise KeyError(f"unknown Config field: {key}")
        lower = raw.lower()
        if lower in {"none", "null"}:
            value = None
        elif isinstance(defaults[key], bool):
            if lower not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"{key} expects a boolean")
            value = lower in {"true", "1", "yes"}
        elif isinstance(defaults[key], int):
            value = int(raw)
        elif isinstance(defaults[key], float) or key in optional_float:
            value = float(raw)
        elif isinstance(defaults[key], str) or key in optional_string:
            value = raw
        else:
            raise TypeError(f"cannot parse override for {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
