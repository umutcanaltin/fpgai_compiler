"""Deterministic offline object-detection evaluation with 101-point AP@IoU.

Inputs MUST be decoded, post-NMS bounding boxes in normalized xyxy format:
 predictions[i] = (K,6) [x1,y1,x2,y2,confidence,class_id];
 ground_truth[i] = (M,5) [x1,y1,x2,y2,class_id].
This module does not infer a YOLO head, decode logits, or run on an FPGA.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


def _check_boxes(items, *, predicted: bool, classes: int) -> np.ndarray:
    width = 6 if predicted else 5
    a = np.asarray(items, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != width or not np.all(np.isfinite(a)):
        raise ValueError('Expected finite decoded bounding-box rows')
    coords = a[:, :4]
    if (np.any(coords < 0) or np.any(coords > 1) or
        np.any(coords[:, 0] >= coords[:, 2]) or
        np.any(coords[:, 1] >= coords[:, 3])):
        raise ValueError('Expected positive-area xyxy boxes normalized to [0,1]')
    labels = a[:, 5 if predicted else 4]
    if np.any(labels < 0) or np.any(labels >= classes) or np.any(labels != np.floor(labels)):
        raise ValueError('Invalid class IDs')
    if predicted and (np.any(a[:, 4] < 0) or np.any(a[:, 4] > 1)):
        raise ValueError('Confidence scores must be probabilities in [0,1]')
    return a


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    overlap = max(0.0, right-left) * max(0.0, bottom-top)
    area_a = float((a[2]-a[0])*(a[3]-a[1]))
    area_b = float((b[2]-b[0])*(b[3]-b[1]))
    return overlap / (area_a + area_b - overlap)


def evaluate_detection(predictions: Sequence, ground_truth: Sequence, *,
                       classes: int, iou_threshold: float = 0.5,
                       confidence_threshold: float = 0.5) -> dict:
    if type(classes) is not int or classes < 1:
        raise ValueError('classes must be positive')
    if not (0 < iou_threshold <= 1 and 0 <= confidence_threshold <= 1):
        raise ValueError('Invalid IoU or confidence threshold')
    if not 1 <= len(predictions) == len(ground_truth):
        raise ValueError('Expected equally sized nonempty predicted and labeled image sets')
    pred = [_check_boxes(x, predicted=True, classes=classes) for x in predictions]
    truth = [_check_boxes(x, predicted=False, classes=classes) for x in ground_truth]
    per_class = {}
    with_truth = []
    confusion = {'tp': 0, 'fp': 0, 'fn': 0}
    for cls in range(classes):
        gts = {sample: row[row[:,4] == cls] for sample,row in enumerate(truth)}
        total = sum(len(row) for row in gts.values())
        ranked = []
        for sample,row in enumerate(pred):
            for local_idx,item in enumerate(row):
                if item[5] == cls:
                    ranked.append((float(item[4]), sample, local_idx, item))
        ranked.sort(key=lambda p: (-p[0], p[1], p[2]))
        used = {sample: set() for sample in gts}
        tp = []
        fp = []
        for confidence, sample, _, item in ranked:
            candidates = gts[sample]
            if len(candidates):
                overlaps = [intersection_over_union(item, b) for b in candidates]
                # Same-class maximum IoU, then smallest GT index on ties.
                unmatched = [(iou, idx) for idx, iou in enumerate(overlaps)
                             if idx not in used[sample] and iou >= iou_threshold]
                accepted = bool(unmatched)
                if accepted:
                    # Highest eligible IoU, deterministic first GT on ties.
                    match = max(unmatched, key=lambda pair: (pair[0], -pair[1]))[1]
            else:
                accepted = False
            if accepted:
                used[sample].add(match)
            tp.append(int(accepted))
            fp.append(int(not accepted))
        scores = np.asarray([item[0] for item in ranked], dtype=np.float64)
        true = np.asarray(tp, dtype=np.int64)
        false = np.asarray(fp, dtype=np.int64)
        found = np.cumsum(true)
        count = np.cumsum(false)
        recall = found/total if total else np.zeros(len(found))
        precision = found/np.maximum(found+count, 1)
        # 101-point interpolated AP; class with no labels is excluded, not AP=0.
        if total:
            with_truth.append(cls)
            thresholds = np.linspace(0.0, 1.0, 101)
            ap = float(np.mean([float(np.max(precision[recall >= r])) if np.any(recall >= r) else 0.0
                                for r in thresholds]))
        else:
            ap = None
        chosen = scores >= confidence_threshold
        class_tp = int(np.sum(true[chosen]))
        class_fp = int(np.sum(false[chosen]))
        class_fn = total - class_tp
        confusion['tp'] += class_tp
        confusion['fp'] += class_fp
        confusion['fn'] += class_fn
        per_class[str(cls)] = {'ap': ap, 'ground_truth': total,
                                'decoded_predictions': len(ranked),
                                'true_positives_at_threshold': class_tp,
                                'false_positives_at_threshold': class_fp,
                                'false_negatives_at_threshold': class_fn}
    if not with_truth:
        raise ValueError('Labeled dataset contains no ground-truth boxes')
    total_pred = confusion['tp'] + confusion['fp']
    total_true = confusion['tp'] + confusion['fn']
    map_value = float(np.mean([per_class[str(i)]['ap'] for i in with_truth]))
    return {'schema': 'fpgai-research/offline-detection/v1',
            'evidence': 'offline-decoded-box-evaluation-only',
            'mAP_101point': map_value,
            'mAP50_101point': map_value if iou_threshold == 0.5 else None,
            'mean_ap_iou_threshold': float(iou_threshold),
            'score_threshold': float(confidence_threshold),
            'micro_precision': confusion['tp']/total_pred if total_pred else 0.0,
            'micro_recall': confusion['tp']/total_true if total_true else 0.0,
            'micro_counts': confusion,
            'included_classes': with_truth,
            'per_class': per_class,
            'images': len(pred),
            'conventions': 'Normalized xyxy; scores and labels already decoded and post-NMS; 101-point interpolated AP; IoU>=threshold',
            'not_hardware_evidence': True}
