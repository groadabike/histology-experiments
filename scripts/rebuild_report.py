"""Rebuild a model's tables and figures from its saved predictions, without retraining.

Reads probs_<set>.npz from a results folder and rewrites metrics.csv, metrics.md,
per_class_f1.csv and the figures, using the same code as training. Extra columns
from the existing metrics.csv (C or epoch) and Macenko failure counts are kept.

Usage
    uv run python scripts/rebuild_report.py run_dir=results/resnet model_label=ResNet-50
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from evaluation import PRETTY, log_report, write_report

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="rebuild_report")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    run_dir = root / cfg.run_dir
    files = {p.stem.removeprefix("probs_"): p for p in run_dir.glob("probs_*.npz")}
    if not files:
        raise FileNotFoundError(f"No probs_*.npz in {cfg.run_dir}")
    order = [s for s in PRETTY if s in files] + sorted(set(files) - set(PRETTY))

    old = pd.read_csv(run_dir / "metrics.csv").set_index("set") if (run_dir / "metrics.csv").exists() else None
    extra = {c: old[c].iloc[0] for c in ("C", "epoch") if old is not None and c in old.columns}

    classes, results = None, {}
    for name in order:
        z = np.load(files[name])
        classes = [str(c) for c in z["classes"]]
        failed = z["failed"] if "failed" in z.files and len(z["failed"]) else None
        if failed is None and old is not None and name in old.index and old.loc[name, "macenko_failed"] > 0:
            # older runs did not save the flags; keep the recorded count
            failed = np.arange(len(z["y"])) < int(old.loc[name, "macenko_failed"])
        results[name] = (z["y"], z["probs"], failed)

    metrics, per_class = write_report(
        results, classes,
        model_label=cfg.model_label, fig_prefix=cfg.fig_prefix or run_dir.name,
        out_dir=run_dir, fig_dir=root / cfg.figures_dir, confusion_for=cfg.confusion_for,
        n_bootstrap=cfg.n_bootstrap, seed=cfg.seed, extra=extra,
    )
    log_report(log, metrics, per_class)
    log.info("Rebuilt report in %s; figures prefixed %s_", cfg.run_dir, cfg.fig_prefix or run_dir.name)


if __name__ == "__main__":
    main()
