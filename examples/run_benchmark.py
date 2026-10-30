"""Benchmark runner: synthetic demo or real NIfTI pairs."""
import argparse
from pathlib import Path

import numpy as np

from src.benchmark import benchmark_cases, write_csv, write_summary
from src.qc import save_overlay
from src.segmenters import get_segmenter


def synthetic_cases(n=5, shape=(64, 64, 48), seed=0):
    """Toy cases: a bright ball (label 1) + noise; GT = the ball."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        r = rng.uniform(10, 14)
        c = np.array(shape) / 2 + rng.uniform(-4, 4, size=3)
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        ball = ((xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2) < r ** 2
        image = ball.astype(np.float32) * 100 + rng.normal(0, 8, shape).astype(np.float32)
        labels = ball.astype(np.int16)
        cases.append((f"synth-{i:02d}", image, labels))
    return cases


def nifti_cases(image_paths, label_paths):
    import nibabel as nib
    cases = []
    for ip, lp in zip(image_paths, label_paths):
        img = np.asanyarray(nib.load(ip).dataobj, dtype=np.float32)
        lab = np.asanyarray(nib.load(lp).dataobj, dtype=np.int16)
        cases.append((Path(ip).stem, img, lab))
    return cases


def main():
    ap = argparse.ArgumentParser(description="Segmentation benchmark")
    ap.add_argument("--segmenter", default="dummy",
                    choices=["dummy", "totalsegmentator", "synthseg"])
    ap.add_argument("--cases", type=int, default=5, help="synthetic cases (demo mode)")
    ap.add_argument("--images", nargs="*", default=[], help="real image NIfTIs")
    ap.add_argument("--labels", nargs="*", default=[], help="reference label NIfTIs")
    ap.add_argument("--out", default="out/demo")
    args = ap.parse_args()

    segmenter = get_segmenter(args.segmenter)
    if args.images:
        assert len(args.images) == len(args.labels), "--images/--labels must pair up"
        cases = nifti_cases(args.images, args.labels)
    else:
        cases = synthetic_cases(n=args.cases)

    rows = benchmark_cases(cases, segmenter)
    out = Path(args.out)
    write_csv(rows, out / "metrics.csv")
    write_summary(rows, out / "summary.md")
    qc_dir = out / "qc"
    for case_id, image, labels in cases:
        pred = segmenter.segment(np.asarray(image, dtype=np.float32))
        save_overlay(image, pred, labels, qc_dir / f"{case_id}.png")
    print(f"segmenter={segmenter.name} cases={len(cases)}")
    print(f"wrote {out / 'metrics.csv'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
