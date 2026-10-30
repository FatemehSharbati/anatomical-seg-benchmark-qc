"""Smoke tests on synthetic data — no downloads needed."""
import numpy as np

from src.metrics import dice_score, hausdorff95, volume_similarity, score_labels
from src.segmenters import DummySegmenter, get_segmenter
from src.benchmark import benchmark_cases


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
