# DeepWeeds Lab Day 2 Code

This directory is the completed student copy of the starter pipeline. The
original starter files outside `submissions/` are not modified.

## Data Layout

Run commands from the repository root. The default configuration expects:

```text
images/*.jpg
data/labels/{labels,train_subset0,val_subset0,test_subset0}.csv
```

The code validates all 17,509 filenames, checks pairwise split disjointness, and
refuses to train if an image file is missing.

## Budget Check

Before full training on Kaggle/Colab, measure one epoch for each planned
backbone:

```powershell
python submissions/2A202602415_nguyen_hong_cuong/code/gpu_budget.py --run --seed 0
```

This writes `submissions/2A202602415_nguyen_hong_cuong/gpu_budget.csv` with the
measured one-epoch time and a conservative total estimate.

## Baseline

```powershell
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=T00 backbone=resnet50 seed=0
```

Only final locked runs should enable test export:

```powershell
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=F01 backbone=convnext_tiny seed=0 loss=ls label_smoothing=0.1 aug=randaug mix=cutmix ema_decay=0.999 save_test_predictions=true
```

Repeat the final configuration and the `T00 + I00` baseline for seeds 0, 1, and
2. Select every architecture, training, and inference decision from validation
results only.

## Recommended Backbone Screen

Use the same T00 recipe and seed for `resnet50`, `resnext50`, `convnext_tiny`,
`deit_small`, `swin_tiny`, and one lightweight reference such as
`efficientnet_b0`. Record macro-F1, parameter count, Conv/Linear GMAC estimate,
and properly measured latency.

## Outputs

- `runs/<exp_id>/seed<k>/config.json`
- `runs/<exp_id>/seed<k>/history.csv`
- `runs/<exp_id>/seed<k>/best.pt`
- validation logits and optional test logits
- `submissions/2A202602415_nguyen_hong_cuong/curves/*.png`
- `submissions/2A202602415_nguyen_hong_cuong/predictions/<exp_id>_seed<k>_{val,test}.csv`

The pipeline defaults to `save_test_predictions=false` to protect the held-out
test set.
