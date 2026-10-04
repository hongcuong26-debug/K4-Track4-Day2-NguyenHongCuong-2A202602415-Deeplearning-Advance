# DeepWeeds Lab Day 2 Report

Student: Nguyen Hong Cuong  
ID: 2A202602415  
Fold: official fold 0

Status: the pipeline and data checks are complete. Cells marked `PENDING GPU
RESULT` must be filled only from real run logs, never inferred or copied from
slides or papers.

## 1. Setup and Reproducibility

- Dataset: DeepWeeds, 17,509 RGB images, 9 classes.
- Split files: `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`.
  The split is not changed, and validation is never merged into training.
- Model-selection metric: validation macro-F1. Ties are resolved by the earlier
  validation-selected checkpoint.
- Validation/test preprocessing: resize shorter side to 256, center crop to the
  evaluation size, and ImageNet normalization.
- Baseline recipe T00: 12 epochs, AdamW, backbone LR `1e-4`, head LR `1e-3`,
  weight decay `0.05`, 1 warmup epoch then cosine decay, batch size 64, AMP
  when CUDA is available.
- Screening seed: 0. Final and baseline runs must use seeds 0, 1, and 2.
- Test predictions are disabled by default. `save_test_predictions=true` is used
  only after all decisions are locked from validation.

Library versions and GPU: `PENDING GPU RESULT`.

## 2. GPU Budget

Before full training, run:

```bash
python submissions/2A202602415_nguyen_hong_cuong/code/gpu_budget.py --run --seed 0
```

This measures one epoch for each planned backbone and writes
`submissions/2A202602415_nguyen_hong_cuong/gpu_budget.csv`.

Current planned minimum:

| Stage | Runs |
|---|---:|
| Backbone screen | 6 |
| Training ablation on selected backbone | 11 |
| Final model, 3 seeds | 3 |
| Baseline T00 + I00, 3 seeds | 3 |
| Contingency | 3 |

Decision after timing: `PENDING GPU RESULT`. If the measured total exceeds the
available quota, reduce epochs from 12 to 10, keep ablations on one backbone,
keep ablation seed count at 1, lower image resolution if needed, and record the
change here.

## 3. Data Checks and EDA

Fold 0 has 10,501 training images, 3,501 validation images, and 3,507 test
images. Pairwise filename intersections are empty; the union has exactly 17,509
images; every filename in the CSV files exists in `images/`.

| Label | Class | Train | Validation | Test |
|---:|---|---:|---:|---:|
| 0 | Chinee Apple | 675 | 225 | 226 |
| 1 | Lantana | 637 | 213 | 213 |
| 2 | Parkinsonia | 618 | 206 | 207 |
| 3 | Parthenium | 613 | 204 | 205 |
| 4 | Prickly Acacia | 637 | 212 | 213 |
| 5 | Rubber Vine | 605 | 202 | 202 |
| 6 | Siam Weed | 644 | 215 | 215 |
| 7 | Snake Weed | 609 | 203 | 204 |
| 8 | Negatives | 5,463 | 1,821 | 1,822 |

`Negatives` is about 52 percent of the dataset, so accuracy can hide errors on
weed classes. Macro-F1 is the main selection metric.

## 4. Backbone Comparison

Backbones planned under the same T00 recipe: ResNet-50, ResNeXt-50,
ConvNeXt-Tiny, DeiT-Small, Swin-Tiny, and EfficientNet-B0. This covers a
classic CNN, modern CNNs, transformers, and a lightweight model.

`PENDING GPU RESULT`: insert the `Backbones` sheet table, the macro-F1 versus
latency or parameter-count plot, and the validation-based reason for choosing
one or two backbones for ablation.

## 5. Training Recipe Ablation

Each experiment changes exactly one factor relative to `T00`. The registry
covers initialization, color/RandAugment, label smoothing, focal loss,
class-weighted CE, Mixup, CutMix, and EMA.

`PENDING GPU RESULT`: insert the `Training` sheet table, report validation
macro-F1 deltas versus `T00`, and compare each delta with seed noise. Differences
smaller than the measured standard deviation should be reported as not
distinguishable.

## 6. Inference and Latency

Planned methods on validation:

| ID | Method | Expected cost |
|---|---|---:|
| I00 | 1 view, FP32 | 1 forward |
| I01 | Horizontal-flip TTA | 2 forwards |
| I02 | 5-crop or multi-scale TTA | about K forwards |
| I03 | Logit aggregation versus probability aggregation | same K |
| I04 | Temperature scaling fitted on validation | no extra forward |
| I05 | Probability ensemble | number of models |
| I06 | Conv-BN fusion or FP16 | 1 forward |

Latency must use at least 10 warmup iterations and at least 50 measured
iterations, with CUDA synchronization before and after timing. Report p50, p95,
p99, GPU name, dtype, batch, image size, and whether preprocessing is included.

`PENDING GPU RESULT`: insert the `Inference` and `Latency` tables, the
accuracy-latency scatter plot, and ECE before/after temperature scaling.

## 7. Final Configuration and Test

The final configuration is selected only from validation. Then run the final
configuration and the `T00 + I00` baseline with seeds 0, 1, and 2, using the test
set exactly once per seed.

`PENDING GPU RESULT`: report mean and sample standard deviation for macro-F1,
top-1, balanced accuracy, and ECE. Include per-class precision, recall, and F1,
especially Chinee Apple and Snake Weed. These numbers must match `eval.py
score`.

## 8. Error Analysis

`PENDING GPU RESULT`: include the final confusion matrix, the most confused
class pairs, and representative mistakes. Separate observations from hypotheses
about causes.

## 9. Limitations

- The official fold is random rather than location-based, so test performance
  may be optimistic for deployment on new farms or seasons.
- The 10-15 epoch lab budget is much smaller than the original paper setting.
- One-seed ablations are screening evidence only; final claims need at least
  three seeds.
- The code's GMAC estimate counts Conv and Linear layers, so attention-heavy
  models may be underestimated. Measured latency is the deployment-relevant
  number.

## 10. Verification Commands

```bash
python eval.py score --pred "submissions/2A202602415_nguyen_hong_cuong/predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python eval.py grade --final "submissions/2A202602415_nguyen_hong_cuong/predictions/F01_seed*_test.csv" --baseline "submissions/2A202602415_nguyen_hong_cuong/predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```
