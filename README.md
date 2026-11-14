# Colorectal Histology: Tissue Classification with External Validation

This project trains and compares two approaches for classifying H&E-stained colorectal tissue patches into nine tissue types, then tests both on an independent external dataset. 

**Key result:** _[One sentence once results are in, e.g. "On external data, the foundation-model probe lost X points of balanced accuracy versus Y for the fine-tuned CNN; Macenko stain normalisation recovered Z points."]_

---

## Contents

- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Limitations](#limitations)
- [How to run](#how-to-run)
- [Repository structure](#repository-structure)
- [References](#references)

---

## Data

| Set | Use | Size | Source |
|---|---|---|----|
| NCT-CRC-HE-100K | Training and internal validation | ~100,000 patches (a stratified subset of 3,000 per class is used) | Kather et al., Zenodo |
| CRC-VAL-HE-7K | External test | ~7,000 patches from different patients | Kather et al., Zenodo |

- **Images:** 224 × 224 px H&E patches.
- **Classes (9):** adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).
- **Licence:** see the Zenodo record for terms of use. The data are not redistributed in this repository.

---

## Methods

### Models

1. **Fine-tuned CNN.** ResNet-50 pretrained on ImageNet; all layers frozen except the last residual block and a new classification head. Trained for 3 epochs with AdamW, mixed precision, and flip and colour-jitter augmentation.
2. **Foundation model + linear probe.** [Phikon](https://huggingface.co/owkin/phikon), a ViT-B pathology foundation model pretrained with self-supervision on histology, used as a frozen feature extractor (768-d CLS embedding per patch). A standardised logistic-regression classifier is trained on the cached features.

### Evaluation

- **Internal validation:** stratified 80/20 split of the training subset (fixed seed, identical for both models).
- **External test:** the full CRC-VAL-HE-7K set, never used for training or model selection.
- **Metrics:** balanced accuracy (robust to class imbalance), per-class precision / recall / F1, row-normalised confusion matrices, and expected calibration error (ECE, 15 bins).

### Robustness experiment

The external images are stain-normalised with the Macenko method (via `torchstain`) to a reference training patch, features are re-extracted, and the external evaluation is repeated. Patches where normalisation fails (typically background or near-empty tiles) are left unnormalised and counted.

---

## Results

| Model | Internal bal. acc. | External bal. acc. | Internal ECE | External ECE |
|---|---|---|---|---|
| ResNet-50 (fine-tuned) | _x.xxx_ | _x.xxx_ | _x.xxx_ | _x.xxx_ |
| Phikon + linear probe | _x.xxx_ | _x.xxx_ | _x.xxx_ | _x.xxx_ |
| Phikon + linear probe, Macenko-normalised external set | – | _x.xxx_ | – | _x.xxx_ |

**Confusion matrices**

| Internal | External |
|---|---|
| ![Internal confusion matrix](figs/cm_internal.png) | ![External confusion matrix](figs/cm_external.png) |

**Observations** _(to complete)_

- Which model generalised better, and by how much.
- Which classes failed most externally (e.g. stroma vs smooth muscle, debris vs mucus) and a likely reason.
- Whether stain normalisation helped, hurt or made no difference, and how many patches it failed on.
- Whether models became over- or under-confident on external data.

---

## Limitations

- **Possible patient-level leakage in internal validation.** The training set does not provide patient identifiers, so the internal split can place patches from the same patient on both sides and overstate accuracy. The external set, drawn from different patients, is the more reliable estimate of real-world performance.
- **Patch-level task.** Patches are pre-extracted and labelled; this project does not cover whole-slide tiling, tissue detection or slide-level aggregation.
- **Single external set.** One external cohort cannot represent the full variation across labs, scanners and staining protocols.
- **Subset training.** A 27k-patch subset is used for speed; results with the full dataset may differ.

---

## How to run

**Requirements:** Python 3.10+, a CUDA GPU for feature extraction and fine-tuning (the linear probe runs on CPU).

```bash
pip install -r requirements.txt
```

**1. Download the data** from Zenodo and unzip into `data/`:

```text
data/NCT-CRC-HE-100K/<CLASS>/*.tif
data/CRC-VAL-HE-7K/<CLASS>/*.tif
```

**2. Build the subset and external index**

```bash
python make_subset.py
```

**3. Extract and cache foundation-model features**

```bash
python extract_features.py splits/train.csv feats/train.npy
python extract_features.py splits/external.csv feats/external.npy
python extract_features.py splits/external.csv feats/external_macenko.npy macenko
```

**4. Linear probe and evaluation**

```bash
python probe_eval.py
```

**5. Fine-tune the CNN** (locally, or on a Slurm cluster)

```bash
python finetune_resnet.py
# or
sbatch slurm/train.sh
```

All random seeds are fixed (`random_state=0`) so splits and subsets are reproducible.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── make_subset.py          # stratified training subset + external index
├── extract_features.py     # Phikon feature extraction (optional Macenko normalisation)
├── probe_eval.py           # linear probe, metrics, calibration, confusion matrices
├── finetune_resnet.py      # ResNet-50 fine-tuning and evaluation
├── slurm/
│   └── train.sh            # example Slurm job
├── splits/                 # generated CSVs (not committed)
├── feats/                  # cached features (not committed)
└── figs/                   # result figures
```

---

## References

- Kather, J. N., et al. (2019). Predicting survival from colorectal cancer histology slides using deep learning: a retrospective multicenter study. *PLOS Medicine*, 16(1), e1002730. Dataset: Zenodo, doi:10.5281/zenodo.1214456.
- Filiot, A., et al. (2023). Scaling self-supervised learning for histopathology with masked image modeling. *medRxiv*. Model: [owkin/phikon](https://huggingface.co/owkin/phikon).
- Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE International Symposium on Biomedical Imaging*, 1107–1110.
- He, K., et al. (2016). Deep residual learning for image recognition. *IEEE CVPR*, 770–778.

---

## Licence

Code released under the MIT licence. Data and pretrained models are subject to their own licences; see the links above.