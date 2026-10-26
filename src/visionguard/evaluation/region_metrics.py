"""Finite-threshold approximation of normalized per-region overlap AUC at FPR <= .3."""
import numpy as np
from scipy.ndimage import label


def aupro(masks, maps, max_fpr=.3, threshold_count=64):
    regions, count = [], 0
    for mask in masks:
        components, n = label(np.asarray(mask) > 0, structure=np.ones((3, 3), dtype=np.uint8))
        count += n
        sizes = np.bincount(components.ravel())[1:]
        regions.append((components, sizes))
    scores = np.concatenate([np.asarray(m).ravel() for m in maps])
    normal = np.concatenate([(np.asarray(mask).ravel() == 0) for mask in masks])
    if not count or not normal.any():
        return None
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, threshold_count)))
    curve = [(0., 0.)]
    for threshold in thresholds[::-1]:
        fpr = float(np.count_nonzero(scores[normal] >= threshold) / np.count_nonzero(normal))
        overlap = 0.
        for score_map, (components, sizes) in zip(maps, regions):
            if len(sizes):
                hit = np.bincount(components.ravel(), weights=(np.asarray(score_map).ravel() >= threshold), minlength=len(sizes)+1)[1:]
                overlap += float(np.sum(hit / sizes))
        curve.append((fpr, overlap / count))
    # Preserve vertical segments. Collapsing repeated FPRs before interpolation
    # would incorrectly give reversed predictions a positive low-FPR area.
    x = np.array([point[0] for point in curve])
    y = np.array([point[1] for point in curve])
    boundary = float(np.interp(max_fpr, x, y))
    keep = x < max_fpr
    return float(np.trapezoid(np.r_[y[keep], boundary], np.r_[x[keep], max_fpr]) / max_fpr)
