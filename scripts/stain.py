"""Shared stain helpers: Macenko normalisation and a controlled stain shift.

Used by extract_features.py and plot_stain_example.py so both apply exactly
the same operations with the same parameters.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def choose_reference(root: Path, train_csv: str, reference: str | None, label: str) -> Path:
    """Return the reference patch: the configured one, or the first *training* patch of `label`.

    Validation patches are excluded so no evaluation data influences preprocessing.
    """
    if reference:
        return root / reference
    df = pd.read_csv(root / train_csv)
    if "split" in df.columns:
        df = df[df.split == "train"]
    candidates = df[df.label == label].sort_values("path")
    if candidates.empty:
        raise ValueError(f"No training patch with label {label}")
    return root / candidates.path.iloc[0]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class Macenko:
    """Macenko normaliser fitted to one reference patch.

    normalise() returns (image, failed). On failure (typically blank background
    patches with too little stain to estimate) the original image is returned.
    """

    def __init__(self, reference: np.ndarray):
        import torchstain

        self._normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        self._normalizer.fit(reference)

    def normalise(self, img: Image.Image) -> tuple[Image.Image, bool]:
        try:
            # Failures raise numpy warnings; they are reported via the flag instead.
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = self._normalizer.normalize(I=np.asarray(img), stains=False)[0]
            if not np.all(np.isfinite(out)):
                raise ValueError("non-finite output")
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), False
        except Exception:
            return img, True


# Stain colour vectors (Ruifrok & Johnston, 2001), as used by scikit-image:
# rows = haematoxylin, eosin, DAB; columns = optical density in R, G, B.
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


class StainShift:
    """Fixed haematoxylin/eosin shift, simulating a lab with different staining.

    Each pixel is converted to optical density and separated into stain
    concentrations (colour deconvolution); each stain is scaled and offset, and
    the image is recombined. Concentrations are not clipped, so with scale 1 and
    bias 0 the image comes back unchanged. The same shift is applied to every patch.
    """

    def __init__(self, h_scale: float, e_scale: float, h_bias: float = 0.0, e_bias: float = 0.0):
        self.h_scale, self.e_scale = h_scale, e_scale
        self.h_bias, self.e_bias = h_bias, e_bias

    def apply(self, img: Image.Image) -> Image.Image:
        rgb = np.maximum(np.asarray(img, dtype=np.float64) / 255.0, 1e-6)
        hed = -np.log(rgb) @ HED_FROM_RGB          # stain concentrations
        hed[..., 0] = hed[..., 0] * self.h_scale + self.h_bias
        hed[..., 1] = hed[..., 1] * self.e_scale + self.e_bias
        out = np.exp(-(hed @ RGB_FROM_HED))        # back to RGB
        return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))

    def describe(self) -> str:
        return (f"h_scale={self.h_scale}, e_scale={self.e_scale}, "
                f"h_bias={self.h_bias}, e_bias={self.e_bias}")


def apply_ops(img: Image.Image, ops, shift: StainShift | None, macenko: Macenko | None):
    """Apply stain operations in order ("shift", "macenko"). Returns (image, macenko_failed)."""
    failed = False
    for op in ops:
        if op == "shift":
            img = shift.apply(img)
        elif op == "macenko":
            img, failed = macenko.normalise(img)
        else:
            raise ValueError(f"Unknown stain operation: {op}")
    return img, failed
