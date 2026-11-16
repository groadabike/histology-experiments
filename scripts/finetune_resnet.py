"""Fine-tune an ImageNet ResNet-50 on histology patches and evaluate it like the probe.

Training uses the same subset and train/val split as the Phikon probe. Only the
last residual block and a new classification head are trained; the frozen layers
also keep their BatchNorm statistics fixed. The epoch with the best internal
validation balanced accuracy is kept (no external data is used for any choice).

The model is then scored on internal validation and the four external conditions
(original, Macenko, stain-shifted, shifted + Macenko), built on the fly from the
images with the same stain settings as the probe (conf/stain.yaml).

Outputs (same format as the probe)
    results/resnet/metrics.csv, metrics.md, per_class_f1.csv, probs_<set>.npz
    results/resnet/model.pt          best weights
    results/resnet/training_log.csv  loss and validation score per epoch
    figs/<run>_cm_<set>.png, figs/<run>_reliability.png   (<run> = output folder name)

Usage
    uv run python scripts/finetune_resnet.py
    uv run python scripts/finetune_resnet.py debug.max_train_batches=20 debug.eval_samples=500
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from evaluation import log_report, write_report
from stain import Macenko, StainShift, apply_ops, choose_reference, load_rgb

log = logging.getLogger(__name__)
# Silence a harmless note printed by every DataLoader worker on macOS.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


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
    """Patches with integer labels; optional stain operations before the transform."""

    def __init__(self, paths, labels, root: Path, transform, ops=(), reference=None, shift_params=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.root = root
        self.transform = transform
        self.ops = list(ops)
        self.reference = reference
        self._shift = StainShift(**shift_params) if "shift" in self.ops else None
        self._macenko = None  # built lazily inside each worker process

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        if "macenko" in self.ops and self._macenko is None:
            self._macenko = Macenko(self.reference)
        img, failed = apply_ops(img, self.ops, self._shift, self._macenko)
        return self.transform(img), int(self.labels[i]), failed


def build_model(cfg: DictConfig, n_classes: int) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if cfg.model.pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    trainable = set(cfg.model.trainable)
    for name, param in model.named_parameters():
        param.requires_grad = name.split(".")[0] in trainable
    return model


def set_train_mode(model: nn.Module, trainable) -> None:
    """Train mode for trainable blocks only; frozen blocks keep fixed BatchNorm statistics."""
    model.train()
    for name, module in model.named_children():
        if name not in trainable:
            module.eval()


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, desc: str):
    model.eval()
    probs, ys, failed = [], [], []
    for x, y, fail in tqdm(loader, desc=desc, unit="batch", leave=False):
        logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
        failed.append(fail.numpy())
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(failed).astype(bool)


def make_loader(dataset, batch_size, shuffle, cfg, device, seed=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


@hydra.main(version_base="1.3", config_path="../conf", config_name="resnet")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    out_dir, fig_dir = root / cfg.output_dir, root / cfg.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)

    # ---- data ---------------------------------------------------------------
    df = pd.read_csv(root / cfg.train_csv)
    external = pd.read_csv(root / cfg.external_csv)
    classes = sorted(df.label.unique())
    lookup = {c: i for i, c in enumerate(classes)}
    df["y"] = df.label.map(lookup)
    external["y"] = external.label.map(lookup)
    if external.y.isna().any():
        raise ValueError(f"External labels not in training: {sorted(set(external.label) - set(classes))}")
    if cfg.debug.eval_samples:
        n = cfg.debug.eval_samples
        val_df = df[df.split == "val"].sample(min(n, (df.split == "val").sum()), random_state=cfg.seed)
        external = external.sample(min(n, len(external)), random_state=cfg.seed)
    else:
        val_df = df[df.split == "val"]
    train_df = df[df.split == "train"]
    log.info("Device: %s | %d train, %d internal val, %d external | classes: %s",
             device, len(train_df), len(val_df), len(external), classes)

    normalise = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    aug = []
    if cfg.augment.flips:
        aug += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if cfg.augment.color_jitter:
        aug += [T.ColorJitter(**cfg.augment.jitter)]
    train_tf, eval_tf = T.Compose(aug + normalise), T.Compose(normalise)

    train_loader = make_loader(PatchDataset(train_df.path, train_df.y, root, train_tf),
                               cfg.train.batch_size, True, cfg, device, cfg.seed)
    val_loader = make_loader(PatchDataset(val_df.path, val_df.y, root, eval_tf),
                             cfg.eval_batch_size, False, cfg, device)

    # ---- model ----------------------------------------------------------------
    model = build_model(cfg, len(classes)).to(device)
    trainable = list(cfg.model.trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    log.info("Trainable parameters: %.1fM of %.1fM",
             sum(p.numel() for p in params) / 1e6, sum(p.numel() for p in model.parameters()) / 1e6)
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = min(len(train_loader), cfg.debug.max_train_batches or len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss()

    # ---- training: keep the epoch with the best internal validation score -------
    history, best = [], None
    for epoch in range(1, cfg.train.epochs + 1):
        set_train_mode(model, trainable)
        start, total_loss, seen = time.time(), 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}", unit="batch", total=steps_per_epoch)
        for step, (x, y, _) in enumerate(bar):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            seen += len(y)
            bar.set_postfix(loss=f"{total_loss / seen:.4f}")
        train_time = time.time() - start

        probs, y_val, _ = predict(model, val_loader, device, "internal val")
        val_bal_acc = balanced_accuracy_score(y_val, probs.argmax(1))
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "val_balanced_accuracy": val_bal_acc, "train_seconds": train_time})
        log.info("Epoch %d: train loss %.4f | internal val balanced accuracy %.4f | %.0fs (%.0f patches/s)",
                 epoch, total_loss / seen, val_bal_acc, train_time, seen / max(train_time, 1e-9))
        if best is None or val_bal_acc > best[0]:
            best = (val_bal_acc, epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))

    best_score, best_epoch, best_state = best
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False, float_format="%.4f")
    log.info("Selected epoch %d (internal val balanced accuracy %.4f)", best_epoch, best_score)

    # ---- evaluation ----------------------------------------------------------------
    reference = None
    if any("macenko" in c.ops for c in cfg.conditions):
        ref_file = choose_reference(root, cfg.train_csv, cfg.stain.reference, cfg.stain.reference_label)
        reference = load_rgb(ref_file)
        log.info("Macenko reference patch: %s", ref_file.relative_to(root))
    shift_params = OmegaConf.to_container(cfg.stain.shift)

    probs, y_val, _ = predict(model, val_loader, device, "internal val")  # with the best weights
    results = {"internal_val": (y_val, probs, None)}
    for cond in cfg.conditions:
        ds = PatchDataset(external.path, external.y, root, eval_tf, cond.ops, reference, shift_params)
        probs, y, failed = predict(model, make_loader(ds, cfg.eval_batch_size, False, cfg, device),
                                   device, cond.name)
        results[cond.name] = (y, probs, failed if "macenko" in cond.ops else None)

    metrics, per_class = write_report(
        results, classes,
        model_label=cfg.model_label, fig_prefix=cfg.fig_prefix or Path(cfg.output_dir).name,
        out_dir=out_dir, fig_dir=fig_dir, confusion_for=cfg.confusion_for,
        n_bootstrap=cfg.n_bootstrap, seed=cfg.seed, extra={"epoch": best_epoch},
    )
    log_report(log, metrics, per_class)
    log.info("Saved tables and model to %s, figures to %s",
             out_dir.relative_to(root), fig_dir.relative_to(root))


if __name__ == "__main__":
    main()
