"""Extract and cache foundation-model features for histology patches.

Runs a frozen pathology foundation model (Phikon by default) once over each
patch list and saves the CLS embedding per patch. Every later experiment then
works on these cached features instead of re-running the model.

Each job writes one .npz file containing:
    features      float32 array, shape (n_patches, 768)
    labels        class name per patch
    paths         image path per patch (same order as the input CSV)
    split         "train"/"val" for the training CSV, empty otherwise
    stain_failed  True where Macenko normalisation failed and the original was kept
    model, ops, reference, shift   metadata for reproducibility

Usage
    uv run python scripts/extract_features.py
    uv run python scripts/extract_features.py 'only=[external_shifted,external_shifted_macenko]'
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel

from stain import Macenko, StainShift, apply_ops, choose_reference, load_rgb

log = logging.getLogger(__name__)
# Silence a harmless note printed by every DataLoader worker on macOS.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


def pick_device(name: str) -> torch.device:
    """auto -> Apple GPU (mps) if available, then CUDA, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PatchDataset(Dataset):
    """Loads patches, applies stain operations, then the model's preprocessing."""

    def __init__(self, paths, root: Path, processor, ops, reference: np.ndarray | None, shift_params):
        self.paths = list(paths)
        self.root = root
        self.processor = processor
        self.ops = list(ops)
        self.reference = reference
        self.shift_params = dict(shift_params)
        self._macenko = None  # built lazily inside each worker process
        self._shift = StainShift(**self.shift_params) if "shift" in self.ops else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        if "macenko" in self.ops and self._macenko is None:
            self._macenko = Macenko(self.reference)
        img, failed = apply_ops(img, self.ops, self._shift, self._macenko)
        pixels = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixels, failed


@hydra.main(version_base="1.3", config_path="../conf", config_name="features")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    device = pick_device(cfg.device)
    log.info("Device: %s | model: %s", device, cfg.model.name)

    jobs = [j for j in cfg.jobs if cfg.only is None or j.name in cfg.only]
    pending = [j for j in jobs if cfg.overwrite or not (root / j.output).exists()]
    for j in jobs:
        if j not in pending:
            log.info("Skipping %s: %s exists (set overwrite=true to redo)", j.name, j.output)
    if not pending:
        return

    processor = AutoImageProcessor.from_pretrained(cfg.model.name)
    model = ViTModel.from_pretrained(cfg.model.name, add_pooling_layer=False).to(device).eval()

    reference, reference_path = None, ""
    if any("macenko" in j.ops for j in pending):
        train_csv = next(j.csv for j in cfg.jobs if j.name == "train")
        ref_file = choose_reference(root, train_csv, cfg.stain.reference, cfg.stain.reference_label)
        reference = load_rgb(ref_file)
        reference_path = str(ref_file.relative_to(root))
        log.info("Macenko reference patch: %s", reference_path)

    for job in pending:
        df = pd.read_csv(root / job.csv)
        shift_params = OmegaConf.to_container(cfg.stain.shift)
        dataset = PatchDataset(df.path, root, processor, job.ops, reference, shift_params)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,  # keep the CSV order so features line up with labels
            num_workers=cfg.num_workers,
            persistent_workers=False,
        )

        features, failed = [], []
        start = time.time()
        with torch.inference_mode():
            for pixels, fail in tqdm(loader, desc=job.name, unit="batch"):
                out = model(pixel_values=pixels.to(device)).last_hidden_state[:, 0]
                features.append(out.float().cpu().numpy())
                failed.append(fail.numpy())

        features = np.concatenate(features)
        failed = np.concatenate(failed).astype(bool)
        out_path = root / job.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            features=features,
            labels=df.label.to_numpy(dtype=str),
            paths=df.path.to_numpy(dtype=str),
            split=df["split"].to_numpy(dtype=str) if "split" in df.columns else np.array([], dtype=str),
            stain_failed=failed,
            model=np.array(cfg.model.name),
            ops=np.array(list(job.ops), dtype=str),
            reference=np.array(reference_path if "macenko" in job.ops else ""),
            shift=np.array(StainShift(**shift_params).describe() if "shift" in job.ops else ""),
        )

        elapsed = time.time() - start
        log.info(
            "%s: %d patches -> %s %s in %.0fs (%.0f patches/s)",
            job.name, len(df), job.output, features.shape, elapsed, len(df) / max(elapsed, 1e-9),
        )
        if "macenko" in job.ops:
            log.info("%s: Macenko failed on %d of %d patches (originals kept)",
                     job.name, failed.sum(), len(failed))


if __name__ == "__main__":
    main()
