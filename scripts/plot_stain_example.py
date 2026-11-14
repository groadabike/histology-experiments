"""Plot patches under each stain condition, one row per class.

Available columns (set in conf/plot_stain.yaml):
    training         a training patch of the same class (the colours the model learned)
    original         an external patch as released
    macenko          the external patch after Macenko normalisation
    shifted          the external patch after the simulated stain shift
    shifted_macenko  the shifted patch after Macenko normalisation

The reference patch used for Macenko is shown in the top-left corner.

Usage
    uv run python scripts/plot_stain_example.py
    uv run python scripts/plot_stain_example.py 'columns=[original,macenko]' seed=3
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")  # write files only, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from PIL import Image

from stain import Macenko, StainShift, apply_ops, choose_reference, load_rgb

log = logging.getLogger(__name__)

TITLES = {
    "training": "Training example",
    "original": "External: original",
    "macenko": "External: Macenko",
    "shifted": "External: stain-shifted",
    "shifted_macenko": "Shifted + Macenko",
}
OPS = {"original": [], "macenko": ["macenko"], "shifted": ["shift"], "shifted_macenko": ["shift", "macenko"]}


@hydra.main(version_base="1.3", config_path="../conf", config_name="plot_stain")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    columns = list(cfg.columns)
    unknown = set(columns) - set(TITLES)
    if unknown:
        raise ValueError(f"Unknown columns {sorted(unknown)}; choose from {list(TITLES)}")

    train = pd.read_csv(root / cfg.train_csv)
    if "split" in train.columns:
        train = train[train.split == "train"]
    external = pd.read_csv(root / cfg.external_csv)

    ref_path = choose_reference(root, cfg.train_csv, cfg.stain.reference, cfg.stain.reference_label)
    reference = load_rgb(ref_path)
    macenko = Macenko(reference)
    shift = StainShift(**cfg.stain.shift)
    log.info("Reference patch: %s | shift: %s", ref_path.relative_to(root), shift.describe())

    classes = list(cfg.classes)
    n_cols = max(len(columns), 1)
    fig, axes = plt.subplots(
        len(classes) + 1, n_cols, figsize=(2.5 * n_cols, 2.6 * (len(classes) + 1)), squeeze=False
    )

    # Top row: the Macenko reference patch.
    axes[0, 0].imshow(reference)
    axes[0, 0].set_title(f"Macenko reference ({cfg.stain.reference_label})", fontsize=10)
    for ax in axes[0]:
        ax.axis("off")

    for row, label in enumerate(classes, start=1):
        train_rows = train[train.label == label]
        ext_rows = external[external.label == label]
        if train_rows.empty or ext_rows.empty:
            raise ValueError(f"No patches for class {label} in one of the CSVs")
        ext_path = ext_rows.sample(1, random_state=cfg.seed).path.iloc[0]
        ext_img = Image.open(root / ext_path).convert("RGB")

        for col, name in enumerate(columns):
            failed = False
            if name == "training":
                path = train_rows.sample(1, random_state=cfg.seed).path.iloc[0]
                img = Image.open(root / path).convert("RGB")
            else:
                img, failed = apply_ops(ext_img, OPS[name], shift, macenko)
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 1:
                ax.set_title(TITLES[name], fontsize=10)
            if failed:
                ax.set_xlabel("Macenko failed:\nunnormalised shown", fontsize=8, color="crimson")
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=0, labelpad=25, va="center")
        log.info("%s: external patch %s", label, ext_path)

    fig.tight_layout()
    out = root / cfg.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    log.info("Saved %s", out.relative_to(root))


if __name__ == "__main__":
    main()
