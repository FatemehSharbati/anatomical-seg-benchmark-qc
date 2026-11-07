"""Smoke tests on synthetic data — no downloads needed."""
import csv

import numpy as np

from src.metrics import dice_score, hausdorff95, volume_similarity, score_labels
from src.segmenters import DummySegmenter, get_segmenter
from src.benchmark import benchmark_cases, write_csv, write_summary
from src.qc import save_overlay


def _ball(shape=(40, 40, 32), r=10, seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    c = np.array(shape) / 2
    return (((xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2) < r ** 2)


def test_dice_perfect():
    m = _ball()
    assert dice_score(m, m) == 1.0


def test_dice_empty():
    z = np.zeros((8, 8, 8), dtype=bool)
    assert dice_score(z, z) == 1.0
    assert dice_score(z, _ball((8, 8, 8), r=2)) == 0.0


def test_hd95_identical_is_zero():
    m = _ball()
    assert hausdorff95(m, m) < 1e-6


def test_hd95_empty_is_nan():
    z = np.zeros((8, 8, 8), dtype=bool)
    assert np.isnan(hausdorff95(z, _ball((8, 8, 8), r=2)))


def test_volume_similarity():
    m = _ball()
    assert volume_similarity(m, m) == 1.0
    assert 0.0 <= volume_similarity(m, np.zeros_like(m)) <= 1.0


def test_score_labels_multilabel():
    lab = np.zeros((16, 16, 16), dtype=np.int16)
    lab[2:6, 2:6, 2:6] = 1
    lab[10:14, 10:14, 10:14] = 2
    scores = score_labels(lab, lab)
    assert set(scores) == {1, 2}
    assert scores[1]["dice"] == 1.0


def test_benchmark_dummy():
    rng = np.random.default_rng(1)
    shape = (48, 48, 36)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ball = ((xx - 24) ** 2 + (yy - 24) ** 2 + (zz - 18) ** 2) < 100
    image = ball.astype(np.float32) * 100 + rng.normal(0, 5, shape).astype(np.float32)
    labels = ball.astype(np.int16)
    rows = benchmark_cases([("c0", image, labels)], DummySegmenter(threshold=50.0))
    assert len(rows) == 1
    assert rows[0]["dice"] > 0.9, rows[0]


def test_get_segmenter():
    assert get_segmenter("dummy").name == "dummy"
    try:
        get_segmenter("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_save_overlay_writes_png(tmp_path):
    # overlay QC was untested; run it on a small synthetic volume and
    # check a valid PNG lands on disk (parent dirs created on demand)
    rng = np.random.default_rng(2)
    shape = (32, 32, 24)
    image = rng.normal(50, 10, shape).astype(np.float32)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ball = ((xx - 16) ** 2 + (yy - 16) ** 2 + (zz - 12) ** 2) < 64
    out = tmp_path / "nested" / "qc" / "overlay.png"
    result = save_overlay(image, ball, ball, str(out), n_slices=3)
    assert result.exists()
    assert result.stat().st_size > 0
    with open(result, "rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n", "output is not a PNG"


def _summary_rows():
    # label 1: hd95 NaN on one case; label 2: hd95 NaN on every case
    return [
        {"case": "c0", "label": 1, "dice": 0.9, "hd95": 1.5, "vol_sim": 0.95},
        {"case": "c1", "label": 1, "dice": 0.8, "hd95": float("nan"), "vol_sim": 0.9},
        {"case": "c0", "label": 2, "dice": 1.0, "hd95": float("nan"), "vol_sim": 1.0},
    ]


def test_write_csv_roundtrip(tmp_path):
    # write_csv was the only untested writer the example runner relies on
    out_csv = tmp_path / "nested" / "metrics.csv"
    result = write_csv(_summary_rows(), out_csv)
    assert result.exists()
    with open(result, newline="") as f:
        saved = list(csv.DictReader(f))
    assert [r["case"] for r in saved] == ["c0", "c1", "c0"]
    assert saved[0]["label"] == "1" and float(saved[0]["dice"]) == 0.9
    assert saved[1]["hd95"] == "nan"


def test_write_summary_drops_nan(tmp_path):
    # a label with hd95 NaN on every case must show "n/a", not crash the
    # mean/std; a single non-NaN value still reports its mean
    out_md = tmp_path / "summary.md"
    result = write_summary(_summary_rows(), out_md)
    text = result.read_text()
    assert "| 1 | 2 | 0.850±0.071 | 1.500±0.000 | 0.925±0.035 |" in text
    assert "| 2 | 1 | 1.000±0.000 | n/a | 1.000±0.000 |" in text
