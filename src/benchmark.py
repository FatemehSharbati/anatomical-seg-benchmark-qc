"""Run a segmenter over cases and collect per-label metrics."""
import csv
from pathlib import Path

import numpy as np

from .metrics import score_labels


def benchmark_cases(cases, segmenter, spacing=(1.0, 1.0, 1.0)):
    """cases: iterable of (case_id, image_3d, label_3d).

    Returns a list of dict rows: case, label, dice, hd95, vol_sim.
    """
    rows = []
    for case_id, image, labels in cases:
        pred = segmenter.segment(np.asarray(image, dtype=np.float32))
        assert pred.shape == np.asarray(labels).shape, "segmenter changed the shape"
        scores = score_labels(pred, labels, spacing=spacing)
        for label_id, m in scores.items():
            rows.append({"case": case_id, "label": label_id, **m})
    return rows


def write_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "label", "dice", "hd95", "vol_sim"])
        w.writeheader()
        w.writerows(rows)
    return out_csv


def write_summary(rows, out_md):
    """Mean ± std per label across cases -> markdown table."""
    import math
    labels = sorted({r["label"] for r in rows})
    lines = ["# Benchmark summary", "",
             "| label | n | dice (mean±std) | hd95 (mean±std) | vol_sim (mean±std) |",
             "|---|---|---|---|---|"]
    for lab in labels:
        sub = [r for r in rows if r["label"] == lab]
        def ms(key):
            vals = [r[key] for r in sub if r[key] == r[key]]  # drop NaN
            if not vals:
                return "n/a"
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
            return f"{m:.3f}±{s:.3f}"
        lines.append(f"| {lab} | {len(sub)} | {ms('dice')} | {ms('hd95')} | {ms('vol_sim')} |")
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md
