"""Segmenter interface + adapters.

A Segmenter maps a 3D image array -> a 3D integer label array.
Real tools are wrapped in adapters; DummySegmenter makes the harness
testable without any downloads.
"""
import numpy as np


class Segmenter:
    name = "base"

    def segment(self, image):
        """image: 3D float array -> 3D int label array (same shape)."""
        raise NotImplementedError


class DummySegmenter(Segmenter):
    """Threshold-based toy segmenter for the synthetic demo."""

    name = "dummy"

    def __init__(self, threshold=50.0, label=1):
        self.threshold = threshold
        self.label = label

    def segment(self, image):
        image = np.asarray(image)
        out = np.zeros(image.shape, dtype=np.int16)
        out[image > self.threshold] = self.label
        return out


class TotalSegmentatorAdapter(Segmenter):
    """Wraps the TotalSegmentator Python API (weights download on first run)."""

    name = "totalsegmentator"

    def __init__(self, task="total", ml=False):
        self.task = task
        self.ml = ml

    def segment(self, image):
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as e:
            raise RuntimeError(
                "TotalSegmentator is not installed. Run: pip install TotalSegmentator"
            ) from e
        import tempfile
        import nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            inp = f"{td}/in.nii.gz"
            outp = f"{td}/out.nii.gz"
            nib.save(nib.Nifti1Image(np.asarray(image, dtype=np.float32), np.eye(4)), inp)
            totalsegmentator(inp, outp, task=self.task, ml=self.ml, quiet=True)
            seg = nib.load(outp).get_fdata()
        return np.asarray(seg, dtype=np.int16)


class SynthSegAdapter(Segmenter):
    """Documents the FreeSurfer SynthSeg CLI call.

    SynthSeg ships with FreeSurfer >= 7.3.2 as `mri_synthseg`.
    This adapter shells out to it; it is NOT installed by requirements.txt.
    """

    name = "synthseg"

    def __init__(self, threads=4):
        self.threads = threads

    def segment(self, image):
        import subprocess
        import tempfile
        import nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            inp = f"{td}/in.nii.gz"
            outp = f"{td}/out.nii.gz"
            nib.save(nib.Nifti1Image(np.asarray(image, dtype=np.float32), np.eye(4)), inp)
            cmd = ["mri_synthseg", "--i", inp, "--o", outp, "--threads", str(self.threads)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    "mri_synthseg failed. Is FreeSurfer >= 7.3.2 on PATH?\n" + r.stderr[-2000:]
                )
            seg = nib.load(outp).get_fdata()
        return np.asarray(seg, dtype=np.int16)


def get_segmenter(name, **kwargs):
    table = {
        "dummy": DummySegmenter,
        "totalsegmentator": TotalSegmentatorAdapter,
        "synthseg": SynthSegAdapter,
    }
    if name not in table:
        raise ValueError(f"unknown segmenter {name!r}; choose from {sorted(table)}")
    return table[name](**kwargs)
