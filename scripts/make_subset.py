"""Build the training subset, internal split and external index.

Outputs (in paths.splits_dir):
    train.csv     path, label, split  (split is "train" or "val")
    external.csv  path, label
    summary.csv   patches per class in each split

Paths are stored relative to the project root, so the CSVs work on any machine
that has the data in the same place.

Usage
    uv run python scripts/make_subset.py
    uv run python scripts/make_subset.py subset.per_class=1000
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def index_dataset(dataset_dir: Path, root: Path, extensions: set[str]) -> pd.DataFrame:
    """List every image as (path relative to root, label = parent folder name)."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_dir} not found. Run scripts/download_data.py first."
        )
    rows = [
        (str(p.relative_to(root)), p.parent.name)
        for p in sorted(dataset_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")
    ]
    if not rows:
        raise RuntimeError(f"No images found in {dataset_dir}")
    return pd.DataFrame(rows, columns=["path", "label"])


@hydra.main(version_base="1.3", config_path="../conf", config_name="subset")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    data_dir = root / cfg.paths.data_dir
    splits_dir = root / cfg.paths.splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)
    extensions = {e.lower() for e in cfg.image_extensions}

    train_all = index_dataset(data_dir / cfg.train_set, root, extensions)
    external = index_dataset(data_dir / cfg.external_set, root, extensions)
    log.info("Indexed %d training and %d external images", len(train_all), len(external))

    # Both sets must use the same class names, or evaluation is meaningless.
    train_classes = set(train_all.label)
    external_classes = set(external.label)
    if train_classes != external_classes:
        raise ValueError(
            f"Class mismatch. Only in training: {sorted(train_classes - external_classes)}; "
            f"only in external: {sorted(external_classes - train_classes)}"
        )

    # Stratified subset: the same number of patches per class (fewer if a class is small).
    counts = train_all.label.value_counts()
    short = counts[counts < cfg.subset.per_class]
    if not short.empty:
        log.warning("Classes with fewer than %d patches (all used): %s",
                    cfg.subset.per_class, short.to_dict())
    # Shuffle once with a fixed seed, then keep the first N of each class.
    subset = (
        train_all.sample(frac=1.0, random_state=cfg.subset.seed)
        .groupby("label")
        .head(cfg.subset.per_class)
        .reset_index(drop=True)
    )

    # Fixed internal train/val split, shared by every model that follows.
    train_idx, val_idx = train_test_split(
        subset.index,
        test_size=cfg.subset.val_fraction,
        stratify=subset.label,
        random_state=cfg.subset.seed,
    )
    subset["split"] = "train"
    subset.loc[val_idx, "split"] = "val"
    subset = subset.sort_values(["split", "label", "path"]).reset_index(drop=True)

    subset.to_csv(splits_dir / "train.csv", index=False)
    external.to_csv(splits_dir / "external.csv", index=False)

    summary = pd.concat(
        {
            "train": subset[subset.split == "train"].label.value_counts(),
            "val": subset[subset.split == "val"].label.value_counts(),
            "external": external.label.value_counts(),
        },
        axis=1,
    ).fillna(0).astype(int).sort_index()
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(splits_dir / "summary.csv", index_label="label")
    log.info("Patches per class:\n%s", summary.to_string())
    log.info("Wrote %s", ", ".join(str(p.relative_to(root)) for p in sorted(splits_dir.glob("*.csv"))))


if __name__ == "__main__":
    main()
