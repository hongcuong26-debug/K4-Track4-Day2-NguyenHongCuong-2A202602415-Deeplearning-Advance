# DeepWeeds Lab Day 2

Student: Nguyen Hong Cuong  
ID: 2A202602415

## Status

The code pipeline, fold-0 checks, EDA scaffold, losses, TTA helpers,
calibration, Conv-BN fusion, benchmark code, notebook, workbook template, and
report template are complete and CPU-tested. Real GPU training metrics have not
been generated in this environment, so `report.md` and `results.xlsx` keep
explicit pending cells instead of fabricated numbers.

## Reproduce

Run from the repository root.

Kaggle profile: https://www.kaggle.com/cuong2612

The public replay notebook URL will be added here after the GPU run is saved
and published from this account.

```bash
pip install -r submissions/2A202602415_nguyen_hong_cuong/code/requirements.txt
python submissions/2A202602415_nguyen_hong_cuong/code/test_implementation.py -v
python submissions/2A202602415_nguyen_hong_cuong/code/experiments.py --stage backbones --seed 0 --dry-run
python submissions/2A202602415_nguyen_hong_cuong/code/gpu_budget.py --run --seed 0
```

After the one-epoch budget check, run the planned experiments:

```bash
python submissions/2A202602415_nguyen_hong_cuong/code/experiments.py --stage backbones --seed 0
python submissions/2A202602415_nguyen_hong_cuong/code/experiments.py --stage ablations --backbone BACKBONE_SELECTED_ON_VAL --seed 0
```

Only final locked configurations should export test predictions:

```bash
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=F01 backbone=BACKBONE_SELECTED_ON_VAL seed=0 save_test_predictions=true
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=T00 backbone=resnet50 seed=0 save_test_predictions=true
```

Repeat final and baseline test export for seeds 0, 1, and 2. Select every
backbone, training recipe, inference method, checkpoint, and temperature only
from validation results.

## GPU Budget Plan

Use `code/gpu_budget.py` on the GPU machine before the full campaign. It trains
one epoch for each planned backbone and writes `gpu_budget.csv`.

Default planned run count:

- 6 backbone-screen runs
- 11 one-factor training-ablation runs on the selected backbone
- 6 final/baseline runs for seeds 0, 1, and 2
- 3 contingency runs for failures or repeats

If the measured total exceeds the available GPU quota, reduce in this order:
epochs 12 to 10, ablations on one backbone only, ablation seed count kept at 1,
lower image resolution, then record the reduction in `report.md`.

## Seeds

- Backbone screen and ablations: seed 0.
- Final and baseline: seeds 0, 1, 2.

## Checked Environment

- Python 3.11.9
- PyTorch 2.14.1 CPU in the local test environment
- timm 1.0.30
- GPU: not available in the current environment

Kaggle profile: https://www.kaggle.com/cuong2612

Kaggle replay notebook: PENDING PUBLICATION.
