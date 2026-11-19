#!/usr/bin/env python3
"""Select one global YOLO confidence threshold per model on the validation set."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path


NAMES = [
    "标识标牌(BS)", "反光锥筒(ZT)", "警示桩(JS)", "防眩板(FX)",
    "声屏障(S)", "防撞桶(FZT)", "水马(S)", "护栏(HL)", "龙门架(L)",
    "绿化(LH)", "路灯(LD)", "桩号(ZH)", "监控(J)", "交通诱导屏(JT)",
    "测速抓拍(CS)", "防眩网(FX)", "信号灯(XH)",
]
LARGE_IDS = {0, 7, 9}


def parse_labels(raw: bytes, prediction: bool) -> list[tuple[int, float, float, float, float, float]]:
    boxes = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        confidence = float(parts[5]) if prediction and len(parts) > 5 else 1.0
        boxes.append((cls, x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence))
    return boxes


def load_archive(path: Path) -> tuple[dict[str, list], dict[str, dict[str, list]]]:
    gt: dict[str, list] = {}
    predictions: dict[str, dict[str, list]] = {"yolo11": {}, "yolo26": {}}
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            normalized = member.name.replace("\\", "/")
            if not normalized.endswith(".txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            stem = Path(normalized).stem
            if "/gt/" in f"/{normalized}":
                gt[stem] = parse_labels(handle.read(), prediction=False)
            elif "/yolo11/labels/" in f"/{normalized}":
                predictions["yolo11"][stem] = parse_labels(handle.read(), prediction=True)
            elif "/yolo26/labels/" in f"/{normalized}":
                predictions["yolo26"][stem] = parse_labels(handle.read(), prediction=True)
    if not gt:
        raise RuntimeError("Archive contains no gt/*.txt labels")
    return gt, predictions


def iou(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[1], b[1]), max(a[2], b[2])
    x2, y2 = min(a[3], b[3]), min(a[4], b[4])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[3] - a[1]) * max(0.0, a[4] - a[2])
    area_b = max(0.0, b[3] - b[1]) * max(0.0, b[4] - b[2])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate(gt: dict[str, list], predictions: dict[str, list], threshold: float) -> dict:
    totals = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in range(len(NAMES))}
    for stem, gt_boxes in gt.items():
        pred_boxes = [box for box in predictions.get(stem, []) if box[5] >= threshold]
        for cls in range(len(NAMES)):
            class_gt = [box for box in gt_boxes if box[0] == cls]
            class_pred = [box for box in pred_boxes if box[0] == cls]
            candidates = []
            for pi, pred in enumerate(class_pred):
                for gi, truth in enumerate(class_gt):
                    overlap = iou(pred, truth)
                    if overlap >= 0.5:
                        candidates.append((overlap, pi, gi))
            used_pred, used_gt = set(), set()
            for _, pi, gi in sorted(candidates, reverse=True):
                if pi not in used_pred and gi not in used_gt:
                    used_pred.add(pi)
                    used_gt.add(gi)
            tp = len(used_pred)
            totals[cls]["tp"] += tp
            totals[cls]["fp"] += len(class_pred) - tp
            totals[cls]["fn"] += len(class_gt) - tp

    per_class = []
    for cls, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append({"class_id": cls, "category": NAMES[cls], **counts,
                          "precision": precision, "recall": recall, "f1": f1})
    valid = [row for row in per_class if row["tp"] + row["fn"] > 0]
    small = [row for row in valid if row["class_id"] not in LARGE_IDS]
    tp = sum(row["tp"] for row in per_class)
    fp = sum(row["fp"] for row in per_class)
    fn = sum(row["fn"] for row in per_class)
    return {
        "threshold": threshold,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_f1": sum(row["f1"] for row in valid) / len(valid),
        "small_macro_f1": sum(row["f1"] for row in small) / len(small),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gt, models = load_archive(args.archive)
    thresholds = []
    value = 0.001
    while value <= 0.951:
        thresholds.append(round(value, 6))
        value += args.step

    summary = {"selection_split": "val", "match_iou": 0.5,
               "objective": "small_macro_f1", "models": {}}
    sweep_rows = []
    for model_name, predictions in models.items():
        results = [evaluate(gt, predictions, threshold) for threshold in thresholds]
        best = max(results, key=lambda row: (row["small_macro_f1"], row["macro_f1"],
                                             row["micro_f1"], row["threshold"]))
        summary["models"][model_name] = best
        for row in results:
            sweep_rows.append({key: value for key, value in row.items() if key != "per_class"} | {"model": model_name})

    with (args.output / "confidence_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output / "confidence_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "threshold", "micro_precision",
                                                     "micro_recall", "micro_f1", "macro_f1",
                                                     "small_macro_f1"])
        writer.writeheader()
        writer.writerows(sweep_rows)
    for model_name, result in summary["models"].items():
        print(model_name, f"threshold={result['threshold']:.3f}",
              f"small_macro_f1={result['small_macro_f1']:.4f}",
              f"micro_f1={result['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
