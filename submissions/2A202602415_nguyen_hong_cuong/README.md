# DeepWeeds Lab Day 2

Student: Nguyen Hong Cuong  
ID: 2A202602415

## Completed run

The required final configuration (`F01`) and baseline (`T00`) were trained on
the official fold 0 with seeds 0, 1, and 2 on an NVIDIA T4. The repository
contains 12 validation/test prediction files, six training curves, one final
confusion matrix, per-epoch logs, `eval.py` outputs, and the populated
`results.xlsx` workbook.

| Configuration | Test macro-F1 | Test top-1 | Balanced accuracy | ECE |
|---|---:|---:|---:|---:|
| F01, ConvNeXt-Tiny | 0.9655 +/- 0.0032 | 0.9712 +/- 0.0027 | 0.9769 +/- 0.0019 | 0.0243 +/- 0.0002 |
| T00, ResNet-50 | 0.7988 +/- 0.0093 | 0.8530 +/- 0.0072 | 0.7607 +/- 0.0147 | 0.0260 +/- 0.0033 |

Values are mean +/- population standard deviation over three seeds, as emitted
by `eval.py score`. Full details are in `report.md`, `results.xlsx`, and
`eval_out/`.

## Colab replay

Executed notebook with saved outputs:

https://colab.research.google.com/drive/1sotmWRH3K7R8zslqD0iRLoUPFz2ej1oY

Kaggle profile: https://www.kaggle.com/cuong2612

## Reproduce

Run from the repository root:

```bash
pip install -r submissions/2A202602415_nguyen_hong_cuong/code/requirements.txt
python submissions/2A202602415_nguyen_hong_cuong/code/test_implementation.py -v
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=T00 backbone=resnet50 seed=0 epochs=10 batch_size=64 save_test_predictions=true
python submissions/2A202602415_nguyen_hong_cuong/code/train.py --set exp_id=F01 backbone=convnext_tiny seed=0 epochs=10 batch_size=48 aug=randaug loss=ce_weighted class_weight_beta=0.9999 mix=cutmix mix_alpha=1.0 ema_decay=0.999 save_test_predictions=true
```

Repeat both training commands for seeds 1 and 2. All model choices are made
from validation macro-F1. Test export is enabled only for the locked T00 and
F01 configurations.

Verify the committed predictions:

```bash
python eval.py score --pred "submissions/2A202602415_nguyen_hong_cuong/predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out submissions/2A202602415_nguyen_hong_cuong/eval_out
python eval.py score --pred "submissions/2A202602415_nguyen_hong_cuong/predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out submissions/2A202602415_nguyen_hong_cuong/eval_out
```

## Run environment

- Google Colab, NVIDIA T4
- Python 3.13.15
- PyTorch 2.11.0+cu130
- timm 1.0.29
- Image size 224, AMP enabled, official fold 0 unchanged
- Train/validation/test sizes: 10,501 / 3,501 / 3,507
