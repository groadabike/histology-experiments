"""Linear probe on cached foundation-model features, with internal and external evaluation.

1. Train a standardised logistic-regression classifier on the training features.
   The regularisation strength C is chosen on the internal validation split only;
   no external data is used for any decision.
2. Score internal validation and every external condition (original, Macenko,
   stain-shifted, shifted + Macenko) with the same metrics.

Outputs
    results/probe/metrics.csv      one row per evaluation set
    results/probe/metrics.md       the same table in Markdown, ready for the README
    results/probe/per_class_f1.csv per-class F1 for every set
    results/probe/probs_<set>.npz  predicted probabilities, for later analysis
    figs/probe_cm_<set>.png        row-normalised confusion matrices
    figs/probe_reliability.png     reliability diagram (calibration)

Usage
    uv run python scripts/probe_eval.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import log_report, write_report

log = logging.getLogger(__name__)

def to_index(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"Labels not seen in training: {unknown}")
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[l] for l in labels])


@hydra.main(version_base="1.3", config_path="../conf", config_name="probe")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    out_dir = root / cfg.output_dir
    fig_dir = root / cfg.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- training data and internal split ---------------------------------
    train = np.load(root / cfg.train_features)
    X, labels, split = train["features"], train["labels"], train["split"]
    if len(split) != len(X):
        raise ValueError("Training features have no train/val split; rerun make_subset.py and extraction.")
    classes = sorted(set(labels))
    y = to_index(labels, classes)
    is_train = split == "train"
    X_tr, y_tr, X_val, y_val = X[is_train], y[is_train], X[~is_train], y[~is_train]
    log.info("Features: %d train, %d internal val, dim %d, %d classes",
             len(y_tr), len(y_val), X.shape[1], len(classes))

    # ---- choose C on internal validation only -----------------------------
    best = None
    for C in cfg.probe.C_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=cfg.probe.max_iter))
        clf.fit(X_tr, y_tr)
        score = balanced_accuracy_score(y_val, clf.predict(X_val))
        log.info("C=%-6g internal val balanced accuracy %.4f", C, score)
        if best is None or score > best[0]:
            best = (score, C, clf)
    _, best_C, clf = best
    log.info("Selected C=%g", best_C)

    # ---- evaluation sets ---------------------------------------------------
    eval_sets = {"internal_val": (X_val, y_val, None)}
    for cond in cfg.conditions:
        path = root / cond.file
        if not path.exists():
            log.warning("Skipping %s: %s not found", cond.name, cond.file)
            continue
        z = np.load(path)
        failed = z["stain_failed"] if "stain_failed" in z.files else None
        eval_sets[cond.name] = (z["features"], to_index(z["labels"], classes), failed)

    results = {name: (ye, clf.predict_proba(Xe), failed) for name, (Xe, ye, failed) in eval_sets.items()}
    metrics, per_class = write_report(
        results, classes,
        model_label="Phikon probe", fig_prefix="probe",
        out_dir=out_dir, fig_dir=fig_dir, confusion_for=cfg.confusion_for,
        n_bootstrap=cfg.n_bootstrap, seed=cfg.seed, extra={"C": best_C},
    )
    log_report(log, metrics, per_class)
    log.info("Saved tables to %s and figures to %s",
             out_dir.relative_to(root), fig_dir.relative_to(root))


if __name__ == "__main__":
    main()
