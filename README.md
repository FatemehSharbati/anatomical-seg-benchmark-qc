# Anatomical Segmentation Benchmark & QC

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small, honest benchmarking harness for **anatomical segmentation tools**
(TotalSegmentator, SynthSeg/FreeSurfer) on brain MRI:

- a common `Segmenter` interface with adapters for real tools
- overlap & surface metrics (Dice, HD95, volume similarity) per label
- overlay QC visualizations
- CSV results + a summary report

The harness runs end-to-end on **synthetic data** with a built-in dummy
segmenter, so the metrics, plotting and reporting code is fully testable
without downloading model weights.

## Quickstart

```bash
pip install -r requirements.txt

# synthetic demo: dummy segmenter vs synthetic ground truth
python examples/run_benchmark.py --cases 5 --out out/demo

# with the real TotalSegmentator (install it first: pip install TotalSegmentator)
python examples/run_benchmark.py --segmenter totalsegmentator \
    --images data/*.nii.gz --labels data/*_seg.nii.gz --out out/ts
```

Results in `--out`:
- `metrics.csv` — one row per (case, label): Dice, HD95, volume similarity
- `summary.md` — mean ± std per label
- `qc/*.png` — segmentation overlays

## Modules

| Module | What it does |
|---|---|
| `src/metrics.py` | Dice, Hausdorff95, volume similarity, per-label scoring |
| `src/segmenters.py` | `Segmenter` interface + `DummySegmenter` + adapters for TotalSegmentator / SynthSeg |
| `src/benchmark.py` | Runs a segmenter over cases, writes CSV + summary |
| `src/qc.py` | Overlay PNGs for visual QC |

## Tool adapters

- **TotalSegmentator** ([wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)):
  install with `pip install TotalSegmentator`; the adapter calls its Python API.
  Downloads ~200MB of weights on first run.
- **SynthSeg** ([BBillot/SynthSeg](https://github.com/BBillot/SynthSeg)):
  ships with FreeSurfer ≥ 7.3.2 (`mri_synthseg`); the adapter documents the CLI
  call and parses its output. Not auto-installed by `requirements.txt`.

## Notes

- Metrics are computed against *your* reference labels — the repo ships no
  medical data. Public test volumes: [IXI](https://brain-development.org/ixi-dataset/).
- Educational / research use only, not a clinical tool.

## License

MIT — see `LICENSE`.
