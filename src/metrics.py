"""Overlap and surface metrics for segmentation benchmarking."""
import numpy as np
from scipy.spatial.distance import cdist


def dice_score(pred, target):
    """Dice coefficient for two binary masks. Empty/empty -> 1.0."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _surface_voxels(mask):
    """Voxels of a binary mask that touch the background (6-connectivity)."""
    mask = np.asarray(mask, dtype=bool)
    inner = np.zeros_like(mask)
    inner[1:, :, :] |= mask[1:, :, :] & ~mask[:-1, :, :]
    inner[:-1, :, :] |= mask[:-1, :, :] & ~mask[1:, :, :]
    inner[:, 1:, :] |= mask[:, 1:, :] & ~mask[:, :-1, :]
    inner[:, :-1, :] |= mask[:, :-1, :] & ~mask[:, 1:, :]
    inner[:, :, 1:] |= mask[:, :, 1:] & ~mask[:, :, :-1]
    inner[:, :, :-1] |= mask[:, :, :-1] & ~mask[:, :, 1:]
    return np.argwhere(mask & inner).astype(float)


def hausdorff95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile Hausdorff distance in mm. NaN if either mask is empty."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not pred.any() or not target.any():
        return float("nan")
    spacing = np.asarray(spacing, dtype=float)
    s_pred = _surface_voxels(pred) * spacing
    s_target = _surface_voxels(target) * spacing
    # subsample for speed on large surfaces
    rng = np.random.default_rng(0)
    if len(s_pred) > 2000:
        s_pred = s_pred[rng.choice(len(s_pred), 2000, replace=False)]
    if len(s_target) > 2000:
        s_target = s_target[rng.choice(len(s_target), 2000, replace=False)]
    d1 = cdist(s_pred, s_target).min(axis=1)
    d2 = cdist(s_target, s_pred).min(axis=1)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def volume_similarity(pred, target):
    """1 - |V_pred - V_target| / (V_pred + V_target). Empty/empty -> 1.0."""
    vp = float(np.asarray(pred, dtype=bool).sum())
    vt = float(np.asarray(target, dtype=bool).sum())
    if vp + vt == 0:
        return 1.0
    return float(1.0 - abs(vp - vt) / (vp + vt))


def score_labels(pred_labels, target_labels, spacing=(1.0, 1.0, 1.0)):
    """Score every non-zero label id present in either volume.

    Returns {label_id: {"dice": ..., "hd95": ..., "vol_sim": ...}}.
    """
    pred_labels = np.asarray(pred_labels)
    target_labels = np.asarray(target_labels)
    ids = sorted(set(np.unique(pred_labels).tolist()) | set(np.unique(target_labels).tolist()))
    ids = [i for i in ids if i != 0]
    out = {}
    for i in ids:
        p = pred_labels == i
        t = target_labels == i
        out[int(i)] = {
            "dice": dice_score(p, t),
            "hd95": hausdorff95(p, t, spacing),
            "vol_sim": volume_similarity(p, t),
        }
    return out
