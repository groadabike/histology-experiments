"""Shared evaluation: metrics, bootstrap confidence intervals and figures.

Used by probe_eval.py and finetune_resnet.py, so every model is scored and
reported with exactly the same code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PRETTY = {
    "internal_val": "Internal validation",
    "external": "External",
    "external_macenko": "External + Macenko",
    "external_shifted": "External, stain-shifted",
    "external_shifted_macenko": "Shifted + Macenko",
}


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size."""
    confidence = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric, n: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI over patches. Patches, not patients, are resampled,
    so intervals are optimistic when patches from one patient are correlated."""
    if n <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores[i] = metric(y[idx], pred[idx])
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def summarise(y: np.ndarray, probs: np.ndarray, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Headline metrics for one evaluation set (y as integer class indices)."""
    pred = probs.argmax(1)
    lo, hi = bootstrap_ci(y, pred, balanced_accuracy_score, n_bootstrap, seed)
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "bal_acc_ci_low": lo,
        "bal_acc_ci_high": hi,
        "macro_f1": f1_score(y, pred, average="macro"),
        "ece": expected_calibration_error(probs, y),
        "mean_confidence": float(probs.max(1).mean()),
    }


def per_class_table(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, probs.argmax(1), labels=range(len(classes)), zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
    )


def plot_confusion(y: np.ndarray, probs: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    """Row-normalised confusion matrix: each row shows where one true class went."""
    cm = confusion_matrix(y, probs.argmax(1), labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, out: Path,
                     n_bins: int = 15) -> None:
    """Reliability diagram: accuracy against confidence for each evaluation set."""
    edges = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    for name, (probs, y) in curves.items():
        confidence = probs.max(1)
        correct = probs.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (confidence > lo) & (confidence <= hi)
            if in_bin.sum() >= 10:  # skip nearly empty bins
                xs.append(confidence[in_bin].mean())
                ys.append(correct[in_bin].mean())
        ece = expected_calibration_error(probs, y, n_bins)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=f"{name} (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    classes: list[str],
    *,
    model_label: str,
    fig_prefix: str,
    out_dir: Path,
    fig_dir: Path,
    confusion_for,
    n_bootstrap: int,
    seed: int,
    extra: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every evaluation set and write tables and figures.

    results maps set name -> (y as class indices, predicted probabilities, macenko-failed flags or None).
    Writes metrics.csv, metrics.md, per_class_f1.csv, probs_<set>.npz to out_dir and
    <fig_prefix>_cm_<set>.png, <fig_prefix>_reliability.png to fig_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_class, curves = [], {}, {}
    for name, (y, probs, failed) in results.items():
        row = {"set": name, **(extra or {}), **summarise(y, probs, n_bootstrap, seed)}
        row["macenko_failed"] = int(np.sum(failed)) if failed is not None else 0
        rows.append(row)
        per_class[name] = per_class_table(y, probs, classes)["f1"]
        np.savez(out_dir / f"probs_{name}.npz", probs=probs, y=y, classes=np.array(classes),
                 failed=np.asarray(failed if failed is not None else [], dtype=bool))
        if name == "internal_val" or name in confusion_for:
            plot_confusion(y, probs, classes, f"{model_label}: {PRETTY.get(name, name)}",
                           fig_dir / f"{fig_prefix}_cm_{name}.png")
        if name in ("internal_val", "external", "external_shifted"):
            curves[PRETTY.get(name, name)] = (probs, y)

    plot_reliability(curves, f"{model_label}: calibration", fig_dir / f"{fig_prefix}_reliability.png")

    metrics = pd.DataFrame(rows).set_index("set")
    per_class_df = pd.DataFrame(per_class)
    metrics.to_csv(out_dir / "metrics.csv", float_format="%.6f")
    per_class_df.to_csv(out_dir / "per_class_f1.csv", float_format="%.6f")

    md = ["| Evaluation set | n | Balanced accuracy (95% CI) | Macro F1 | ECE |",
          "|---|---|---|---|---|"]
    for name, r in metrics.iterrows():
        ci = "" if np.isnan(r.bal_acc_ci_low) else f" ({r.bal_acc_ci_low:.3f}–{r.bal_acc_ci_high:.3f})"
        md.append(f"| {PRETTY.get(name, name)} | {int(r.n)} | {r.balanced_accuracy:.3f}{ci} "
                  f"| {r.macro_f1:.3f} | {r.ece:.3f} |")
    (out_dir / "metrics.md").write_text("\n".join(md) + "\n")
    return metrics, per_class_df


def log_report(log, metrics: pd.DataFrame, per_class: pd.DataFrame) -> None:
    cols = ["n", "balanced_accuracy", "bal_acc_ci_low", "bal_acc_ci_high", "macro_f1", "ece", "macenko_failed"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        log.info("Results:\n%s", metrics[cols].to_string())
        log.info("Per-class F1:\n%s", per_class.to_string())
