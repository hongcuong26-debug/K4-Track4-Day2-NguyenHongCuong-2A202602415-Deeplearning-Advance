"""Measure one-epoch backbone cost and estimate the GPU budget.

Run this on Kaggle/Colab or a local CUDA machine before the real experiment
campaign. It deliberately keeps save_test_predictions=False so the test set is
not used during planning.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments import BACKBONES
from train import Config, run, run_dir


def budget_rows(out_dir: str, seed: int, epochs: int, ablation_runs: int,
                final_runs: int, contingency_runs: int) -> list[dict]:
    rows = []
    for index, (screen_id, backbone) in enumerate(BACKBONES.items(), 1):
        cfg = Config(exp_id=f"G{index:02d}_{screen_id}", backbone=backbone, seed=seed,
                     epochs=1, out_dir=out_dir, save_test_predictions=False)
        summary_path = run_dir(cfg) / "summary.json"
        measured = ""
        if summary_path.exists():
            measured = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "mean_epoch_seconds", "")
        rows.append({
            "screen_exp_id": screen_id,
            "budget_exp_id": cfg.exp_id,
            "backbone": backbone,
            "seed": seed,
            "one_epoch_seconds": measured,
            "planned_epochs_per_train": epochs,
            "estimated_one_train_seconds": (
                "" if measured == "" else float(measured) * epochs
            ),
        })

    measured_values = [float(row["one_epoch_seconds"]) for row in rows
                       if row["one_epoch_seconds"] != ""]
    typical = max(measured_values) if measured_values else ""
    total_runs = len(BACKBONES) + ablation_runs + final_runs + contingency_runs
    rows.append({
        "screen_exp_id": "TOTAL",
        "budget_exp_id": "",
        "backbone": "uses slowest measured backbone as conservative estimate",
        "seed": seed,
        "one_epoch_seconds": typical,
        "planned_epochs_per_train": epochs,
        "estimated_one_train_seconds": (
            "" if typical == "" else typical * epochs
        ),
        "planned_train_runs": total_runs,
        "estimated_total_seconds": (
            "" if typical == "" else typical * epochs * total_runs
        ),
        "estimated_total_hours": (
            "" if typical == "" else typical * epochs * total_runs / 3600.0
        ),
    })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "screen_exp_id", "budget_exp_id", "backbone", "seed",
        "one_epoch_seconds", "planned_epochs_per_train",
        "estimated_one_train_seconds", "planned_train_runs",
        "estimated_total_seconds", "estimated_total_hours",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="actually train one epoch for each backbone")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="allow slow CPU timing for debugging only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--ablation-runs", type=int, default=11,
                        help="T00 plus planned one-factor ablations")
    parser.add_argument("--final-runs", type=int, default=6,
                        help="3 seeds final + 3 seeds baseline")
    parser.add_argument("--contingency-runs", type=int, default=3,
                        help="reserve for failed or repeated runs")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--csv", default=(
        "submissions/2A202602415_nguyen_hong_cuong/gpu_budget.csv"))
    args = parser.parse_args()

    if args.run:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={device}")
        if device != "cuda" and not args.allow_cpu:
            raise SystemExit(
                "CUDA GPU was not detected. Run without --run to create the "
                "budget template, or run on Kaggle/Colab/GPU. Use --allow-cpu "
                "only for debugging; CPU timings are not valid GPU budget data."
            )
        for index, (screen_id, backbone) in enumerate(BACKBONES.items(), 1):
            cfg = Config(exp_id=f"G{index:02d}_{screen_id}", backbone=backbone,
                         seed=args.seed, epochs=1, out_dir=args.out_dir,
                         save_test_predictions=False)
            print(f"RUN {cfg.exp_id}: {backbone}")
            print(json.dumps(run(cfg), indent=2))

    rows = budget_rows(args.out_dir, args.seed, args.epochs, args.ablation_runs,
                       args.final_runs, args.contingency_runs)
    write_csv(Path(args.csv), rows)
    print(args.csv)


if __name__ == "__main__":
    main()
