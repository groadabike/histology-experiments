# Colorectal Histology: Tissue Classification with External Validation

How well does a histology image classifier hold up on patients, and stains, it has never seen?

This project compares a linear probe on a pathology foundation model (Phikon) with a fine-tuned ImageNet ResNet-50 for classifying H&E-stained colorectal tissue patches into nine tissue types. Both are tested on an external set from different patients, under a simulated stain shift, and with Macenko stain normalisation. Beyond accuracy, it measures calibration: whether a model's confidence can be trusted when the data change.

[//]: # (**Key result.** On the external set, the two approaches performed equally well &#40;balanced accuracy 0.937 vs 0.935&#41;. Under a simulated stain shift, the frozen foundation-model features lost 8.2 points and became overconfident, while the fine-tuned CNN lost 1.3 points without colour augmentation and 0.2 with it. Per-patch Macenko normalisation lowered accuracy for every model, mainly through errors on debris.)

[//]: # ()
[//]: # (![Balanced accuracy by model and evaluation set]&#40;figs/comparison.png&#41;)

---

## Contents

- [Data](#data)
- [Methods](#methods)

[//]: # (- [Results]&#40;#results&#41;)

[//]: # (- [Limitations]&#40;#limitations&#41;)
- [How to run](#how-to-run)
- [Repository structure](#repository-structure)
- [References](#references)

---

## Data

| Set | Use | Size | Source |
|---|---|---|---|
| NCT-CRC-HE-100K | Training and internal validation | 100,000 patches; a stratified subset of 3,000 per class (27,000) is used, split 80/20 | Kather et al., Zenodo |
| CRC-VAL-HE-7K | External test | 7,180 patches from 50 patients, none shared with the training set | Kather et al., Zenodo |

- **Images:** 224 × 224 px H&E patches.
- **Classes (9):** adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).
- **Colour:** the training set was Macenko-normalised by the dataset authors, and the external patches show little colour difference from it. The stain shift below therefore simulates a lab with genuinely different staining.
- **Licence:** CC BY 4.0 (Kather, Halama & Marx, 2018; see References). The data are not redistributed here..

---

## Methods

### Models

1. **Foundation model + linear probe.** [Phikon](https://huggingface.co/owkin/phikon), a ViT-B/16 pretrained with self-supervision on histology, is used as a frozen feature extractor (768-d CLS embedding per patch). A standardised logistic-regression classifier is trained on the cached features; its regularisation strength (C ∈ {0.01, 0.1, 1}) is chosen on internal validation.
2. **Fine-tuned CNN.** ResNet-50 pretrained on ImageNet. Only the last residual block and a new classification head are trained (15.0M of 23.5M parameters); frozen layers also keep their BatchNorm statistics. Three epochs, AdamW (lr 3 × 10⁻⁴, cosine schedule), flips and mild colour jitter. The epoch with the best internal validation score is kept.
3. **Control: fine-tuned CNN without colour jitter.** Identical to model 2 but with flips only, to separate the effect of fine-tuning from the effect of colour augmentation.

All models use the same training subset and the same train/validation split. No external data is used for any choice.

### Stain conditions

| Condition | What is done to the external patches |
|---|---|
| External | Nothing: patches as released |
| External, stain-shifted | A fixed shift simulating another lab: each patch is separated into haematoxylin and eosin by colour deconvolution (Ruifrok & Johnston), haematoxylin × 1.4 and eosin × 0.6, then recombined |
| External + Macenko | Macenko normalisation of each patch to a reference tumour patch from the training split ([torchstain](https://github.com/EIDOSLAB/torchstain)) |
| Shifted + Macenko | The stain shift, then Macenko normalisation |

![Training, external, stain-shifted and normalised patches](figs/stain_example.png)

### Metrics

- **Balanced accuracy** (mean per-class recall) with a 95% bootstrap confidence interval (1,000 resamples of patches).
- **Macro F1** and per-class F1.
- **Expected calibration error** (ECE, 15 bins): the average gap between a model's confidence and its accuracy.
- Row-normalised confusion matrices and reliability diagrams.

---

[//]: # ()
[//]: # (## Results)

[//]: # ()
[//]: # (| Evaluation set | Phikon probe | ResNet-50 | ResNet-50, no colour jitter |)

[//]: # (|---|---|---|---|)

[//]: # (| Internal validation | 0.998 | 0.990 | 0.991 |)

[//]: # (| External | 0.937 | 0.935 | 0.923 |)

[//]: # (| External, stain-shifted | 0.855 | 0.933 | 0.910 |)

[//]: # (| External + Macenko | 0.861 | 0.891 | 0.887 |)

[//]: # (| Shifted + Macenko | 0.873 | 0.863 | 0.871 |)

[//]: # (| **Change under stain shift &#40;points&#41;** | **−8.2** | **−0.2** | **−1.3** |)

[//]: # (| ECE on stain-shifted data | 0.060 | 0.021 | 0.034 |)

[//]: # ()
[//]: # (Balanced accuracy unless stated. Full metrics, confidence intervals and per-class F1 are in `results/<model>/`.)

[//]: # ()
[//]: # (| Phikon probe, external | ResNet-50, external |)

[//]: # (|---|---|)

[//]: # (| ![Phikon probe confusion matrix, external set]&#40;figs/probe_cm_external.png&#41; | ![ResNet-50 confusion matrix, external set]&#40;figs/resnet_cm_external.png&#41; |)

[//]: # ()
[//]: # (### Findings)

[//]: # ()
[//]: # (1. **Internal validation overstates performance.** Every model scores 0.99 or more internally and 0.92–0.94 externally, and the internal ranking does not carry over. Most external errors are in smooth muscle and stroma &#40;probe F1 0.74 for both&#41;.)

[//]: # (2. **The two approaches generalise equally well on the data as released**, with different error profiles. The ResNet-50 handles smooth muscle better &#40;F1 0.85 vs 0.74&#41;; the probe is stronger on debris, normal mucosa and lymphocytes.)

[//]: # (3. **Frozen foundation-model features are the most sensitive to stain variation.** Under the stain shift the probe loses 8.2 points, mostly on tumour &#40;F1 0.99 → 0.82&#41;, stroma and smooth muscle, and its calibration error nearly triples: it is more often wrong without being less confident. The fine-tuned CNN loses 1.3 points even without colour augmentation, so most of its robustness comes from fine-tuning rather than from augmentation.)

[//]: # (4. **Per-patch Macenko normalisation is not a safe default.** It partly recovers the probe on shifted data &#40;0.855 → 0.873&#41; but lowers accuracy for every model on the unshifted external set, and debris collapses in all three &#40;to F1 0.64–0.68 on shifted data&#41;. Per-patch stain estimates are unreliable on patches without a typical haematoxylin–eosin mix, yet only one patch failed outright, so the damage is silent. Normalising at slide level, where stain vectors are estimated over much more tissue, is the usual remedy.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Limitations)

[//]: # ()
[//]: # (- **One training run per model.** Confidence intervals cover patch sampling only, not training randomness. Differences of about one point &#40;for example, with and without colour jitter&#41; could change with another seed; the 8-point gap is far larger.)

[//]: # (- **Optimistic confidence intervals.** Patches, not patients, are resampled, and patches from one patient are correlated.)

[//]: # (- **Possible patient-level leakage in internal validation.** The training set has no patient identifiers, so the internal split can place patches from the same patient on both sides.)

[//]: # (- **A simulated stain shift.** One global transformation is a controlled test, not a substitute for real data from other labs and scanners.)

[//]: # (- **Patch-level task.** Patches are pre-extracted and labelled; whole-slide tiling, tissue detection and slide-level aggregation are not covered.)

[//]: # (- **Subset training.** 27,000 of 100,000 training patches are used, for speed.)

---

## How to run

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/). The device is chosen automatically: Apple GPU (MPS), then CUDA, then CPU. All settings are [Hydra](https://hydra.cc) configs in `conf/`, and any of them can be overridden from the command line.

```bash
uv sync
```

```bash
# 1. Download and extract the data from Zenodo (resumable, checksum-verified)
uv run python scripts/download_data.py

# 2. Stratified training subset, internal split and external index
uv run python scripts/make_subset.py

# 3. (optional) Look at the stain conditions
uv run python scripts/plot_stain_example.py

# 4. Extract and cache Phikon features for every condition
uv run python scripts/extract_features.py

# 5. Linear probe and evaluation
uv run python scripts/probe_eval.py

# 6. Fine-tune ResNet-50, with and without colour jitter (about 25-30 min each on an Apple M3 Pro)
uv run python scripts/finetune_resnet.py
uv run python scripts/finetune_resnet.py augment.color_jitter=false output_dir=results/resnet_nojitter \
    "model_label='ResNet-50, no colour jitter'"

# 7. Comparison table and figure
uv run python scripts/compare_models.py
```

Useful overrides:

```bash
uv run python scripts/finetune_resnet.py debug.max_train_batches=20 debug.eval_samples=500   # quick smoke test
uv run python scripts/plot_stain_example.py stain.shift.h_scale=1.6 stain.shift.e_scale=0.5  # try another shift
uv run python scripts/rebuild_report.py run_dir=results/resnet model_label=ResNet-50          # redo figures from saved predictions
```

All random seeds are fixed, so subsets, splits and bootstrap intervals are reproducible.

---

## Repository structure

```text
.
├── README.md
├── pyproject.toml / uv.lock
├── conf/                       # Hydra configs, one per script
│   ├── stain.yaml              # Macenko reference and stain shift, shared by all scripts
│   ├── download.yaml  subset.yaml  plot_stain.yaml  features.yaml
│   └── probe.yaml  resnet.yaml  compare.yaml  rebuild_report.yaml
├── scripts/
│   ├── download_data.py        # Zenodo download, resume, md5 check, extraction
│   ├── make_subset.py          # stratified subset, internal split, external index
│   ├── stain.py                # Macenko normalisation and stain shift (shared)
│   ├── plot_stain_example.py   # patches under each stain condition
│   ├── extract_features.py     # cached Phikon features per condition
│   ├── evaluation.py           # metrics, bootstrap CIs, figures (shared)
│   ├── probe_eval.py           # linear probe and evaluation
│   ├── finetune_resnet.py      # ResNet-50 fine-tuning and evaluation
│   ├── rebuild_report.py       # rebuild tables and figures from saved predictions
│   └── compare_models.py       # comparison table and figure
├── results/                    # metrics tables per model (predictions and weights not committed)
├── figs/                       # figures used in this README
├── data/  splits/  feats/      # generated, not committed
```

---

## References


**Data**

- Kather, J. N., Halama, N., & Marx, A. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* (Version v0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1214456 (CC BY 4.0). Contains NCT-CRC-HE-100K and CRC-VAL-HE-7K.
- Kather, J. N., et al. (2019). Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLOS Medicine*, 16(1), e1002730. https://doi.org/10.1371/journal.pmed.1002730

**Models**

- Filiot, A., Ghermi, R., Olivier, A., Jacob, P., Fidon, L., Mac Kain, A., Saillard, C., & Schiratti, J.-B. (2023). Scaling self-supervised learning for histopathology with masked image modeling. *medRxiv*. https://doi.org/10.1101/2023.07.21.23292757. Weights: [owkin/phikon](https://huggingface.co/owkin/phikon).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

**Methods**

- Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE International Symposium on Biomedical Imaging (ISBI)*, 1107–1110. https://doi.org/10.1109/ISBI.2009.5193250

---

## Licence

Code released under the MIT licence. Data and pretrained models are subject to their own licences; see the links above.
