"""Create the required formatted results.xlsx workbook without inventing metrics."""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OUTPUT = Path(__file__).resolve().parent.parent / "results.xlsx"
HEADERS = {
    "Backbones": ["exp_id", "backbone", "weight_tag", "params_M", "GMAC", "resolution",
                  "epochs", "seed", "macro_F1_val", "top1_val", "train_s_per_epoch",
                  "latency_batch1_ms", "notes", "status"],
    "Training": ["exp_id", "backbone", "axis_A_to_G", "difference_from_T00", "seed",
                 "macro_F1_val", "top1_val", "delta_vs_T00", "rare_class_F1", "notes", "status"],
    "Inference": ["exp_id", "method", "model_checkpoint", "K", "macro_F1_val", "top1_val",
                  "ECE_val", "p50_ms_batch1", "p95_ms_batch1", "p99_ms_batch1",
                  "images_per_s", "relative_cost_vs_I00", "status"],
    "Final": ["exp_id", "configuration", "seed", "macro_F1_val", "macro_F1_test",
              "top1_test", "balanced_acc_test", "ECE_test", "mean_plus_minus_std", "status"],
    "PerClass": ["configuration", "class", "test_support", "precision", "recall", "F1", "status"],
    "Latency": ["configuration", "GPU", "dtype", "batch", "resolution", "BN_fused",
                "p50_ms", "p95_ms", "p99_ms", "images_per_s", "warmup", "iterations", "status"],
    "Summary": ["rank", "exp_id", "configuration", "macro_F1_val", "top1_val", "params_M",
                "p95_ms_batch1", "relative_cost", "evidence_note", "status"],
}

BACKBONES = [
    ("B01", "resnet50"), ("B02", "resnext50_32x4d"), ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"), ("B05", "swin_tiny_patch4_window7_224"),
    ("B06", "efficientnet_b0"),
]
TRAINING = [
    ("T00", "baseline", "baseline recipe"),
    ("T01", "A", "init=scratch"), ("T02", "A", "init=frozen"),
    ("T03", "B", "aug=color"), ("T04", "B", "aug=randaug"),
    ("T05", "C", "loss=label_smoothing(0.1)"), ("T06", "C", "loss=focal(gamma=2)"),
    ("T07", "C", "loss=class_weighted(beta=0.9999)"),
    ("T08", "B", "mixup(alpha=0.4)"), ("T09", "B", "cutmix(alpha=1.0)"),
    ("T10", "G", "EMA(decay=0.999)"),
]
INFERENCE = [
    ("I00", "one view", 1), ("I01", "horizontal flip TTA", 2),
    ("I02", "five crop TTA", 5), ("I03", "logit/probability aggregation", 2),
    ("I04", "temperature scaling fitted on val", 1), ("I05", "probability ensemble", 2),
    ("I06", "Conv-BN fusion or FP16", 1),
]


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 30
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    for index, column in enumerate(ws.columns, 1):
        content = [str(cell.value or "") for cell in column]
        width = min(max(max(map(len, content)) + 2, 11), 34)
        ws.column_dimensions[get_column_letter(index)].width = width
    metric_columns = [cell.column for cell in ws[1] if "F1" in str(cell.value) or "top1" in str(cell.value)]
    for column in metric_columns:
        letter = get_column_letter(column)
        ws.conditional_formatting.add(f"{letter}2:{letter}500",
            ColorScaleRule(start_type="min", start_color="FEE2E2", mid_type="percentile",
                           mid_value=50, mid_color="FEF3C7", end_type="max", end_color="DCFCE7"))


def main():
    workbook = Workbook()
    workbook.remove(workbook.active)
    sheets = {name: workbook.create_sheet(name) for name in HEADERS}
    for name, headers in HEADERS.items():
        sheets[name].append(headers)

    for exp_id, backbone in BACKBONES:
        sheets["Backbones"].append([exp_id, backbone, "", "", "", 224, 12, 0,
                                    "", "", "", "", "Awaiting real GPU log", "PLANNED"])
    for exp_id, axis, difference in TRAINING:
        sheets["Training"].append([exp_id, "select from validation", axis, difference, 0,
                                   "", "", "", "", "One-factor comparison", "PLANNED"])
    for exp_id, method, views in INFERENCE:
        sheets["Inference"].append([exp_id, method, "best validation checkpoint", views,
                                    "", "", "", "", "", "", "", "", "PLANNED"])

    summary = sheets["Summary"]
    summary.append(["", "", "No experiment metrics are populated until real GPU runs finish.",
                    "", "", "", "", "", "Select only from validation; test once per final seed.", "PENDING"])
    summary.append(["", "EDA", "Fold 0 split verification", "", "", "", "", "",
                    "train=10501, val=3501, test=3507; disjoint; union=17509", "VERIFIED"])

    for sheet in sheets.values():
        format_sheet(sheet)
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
