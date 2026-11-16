"""Compare models across evaluation sets: one table and one chart for the README.

Reads metrics.csv from each model's results directory and writes
    results/comparison.md   balanced accuracy per set and model, the drop caused by
                            the stain shift, and calibration error on shifted data
    results/comparison.csv  the same numbers, long format
    figs/comparison.png     balanced accuracy with 95% CIs, per model and set

Usage
    uv run python scripts/compare_models.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from evaluation import PRETTY

log = logging.getLogger(__name__)

# Validated categorical palette (colour-blind safe for 3 series), fixed order.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]  # second encoding, so identity never relies on colour alone
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

SHORT = {
    "internal_val": "Internal\nvalidation",
    "external": "External",
    "external_macenko": "External\n+ Macenko",
    "external_shifted": "Stain-\nshifted",
    "external_shifted_macenko": "Shifted\n+ Macenko",
}


def load(cfg: DictConfig, root: Path) -> pd.DataFrame:
    frames = []
    for model in cfg.models:
        path = root / model.dir / "metrics.csv"
        if not path.exists():
            log.warning("Skipping %s: %s not found", model.label, path.relative_to(root))
            continue
        df = pd.read_csv(path)
        df.insert(0, "model", model.label)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No metrics.csv found for any model")
    data = pd.concat(frames, ignore_index=True)
    return data[data["set"].isin(cfg.sets)]


def write_table(data: pd.DataFrame, cfg: DictConfig, models: list[str], out: Path) -> None:
    acc = data.pivot(index="set", columns="model", values="balanced_accuracy").reindex(cfg.sets)[models]
    ece = data.pivot(index="set", columns="model", values="ece")[models]

    header = "| Evaluation set | " + " | ".join(models) + " |"
    lines = [header, "|---|" + "---|" * len(models)]
    for s in cfg.sets:
        if s in acc.index and acc.loc[s].notna().any():
            lines.append(f"| {PRETTY.get(s, s)} | " + " | ".join(f"{v:.3f}" for v in acc.loc[s]) + " |")

    a, b = cfg.shift_drop["from"], cfg.shift_drop["to"]
    if a in acc.index and b in acc.index:
        drop = (acc.loc[b] - acc.loc[a]) * 100
        lines.append("| **Change under stain shift (points)** | "
                     + " | ".join(f"**{v:+.1f}**" for v in drop) + " |")
    if b in ece.index:
        lines.append(f"| ECE on {PRETTY.get(b, b).lower()} data | "
                     + " | ".join(f"{v:.3f}" for v in ece.loc[b]) + " |")

    note = ("\nBalanced accuracy unless stated. 95% CIs (patch-level bootstrap) are in each model's "
            "metrics.csv; they do not include training randomness (one seed per model).\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n" + note)


def plot(data: pd.DataFrame, cfg: DictConfig, models: list[str], out: Path) -> None:
    sets = [s for s in cfg.sets if s in set(data["set"])]
    x = np.arange(len(sets))
    width = 0.22
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Shade the stain-shift group: it carries the main finding.
    shifted = cfg.shift_drop["to"]
    if shifted in sets:
        i = sets.index(shifted)
        ax.axvspan(i - 0.45, i + 0.45, color="#f1f0ec", zorder=0)

    for k, model in enumerate(models):
        d = data[data.model == model].set_index("set").reindex(sets)
        y = d["balanced_accuracy"].to_numpy()
        err = np.vstack([y - d["bal_acc_ci_low"].to_numpy(), d["bal_acc_ci_high"].to_numpy() - y])
        err = np.where(np.isfinite(err), err, 0)
        ax.errorbar(
            x + offsets[k], y, yerr=err, fmt=MARKERS[k], color=COLOURS[k], markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=2, capsize=0, label=model, zorder=3,
        )
        # Direct label on the stain-shifted group, where the models separate.
        if shifted in sets:
            i = sets.index(shifted)
            if np.isfinite(y[i]):
                ax.annotate(f"{y[i]:.3f}", (x[i] + offsets[k], y[i]), xytext=(9, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ax.set_xticks(x, [SHORT.get(s, s) for s in sets], fontsize=9.5, color=INK)
    ax.set_ylabel("Balanced accuracy (95% CI)", fontsize=10, color=INK)
    if cfg.y_limits:
        ax.set_ylim(*cfg.y_limits)
    else:
        lo = np.nanmin(data["bal_acc_ci_low"].fillna(data["balanced_accuracy"]))
        ax.set_ylim(np.floor((lo - 0.01) * 50) / 50, 1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="y", colors=INK_2, length=0, labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.set_title("The frozen foundation-model probe is the most sensitive to stain shift",
                 fontsize=10.5, color=INK, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), frameon=False,
              fontsize=9, handletextpad=0.4, columnspacing=1.4, labelcolor=INK)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../conf", config_name="compare")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    data = load(cfg, root)
    models = [m.label for m in cfg.models if m.label in set(data.model)]
    if len(models) > len(COLOURS):
        raise ValueError(f"At most {len(COLOURS)} models per chart; split the comparison")

    data.to_csv(root / cfg.output_csv, index=False, float_format="%.4f")
    write_table(data, cfg, models, root / cfg.output_table)
    plot(data, cfg, models, root / cfg.output_figure)
    log.info("Comparison table:\n%s", (root / cfg.output_table).read_text())
    log.info("Saved %s, %s and %s", cfg.output_table, cfg.output_csv, cfg.output_figure)


if __name__ == "__main__":
    main()
