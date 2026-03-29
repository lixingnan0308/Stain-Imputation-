# AdSI-MIMO: Adaptive Stain Imputation with Multi-Input and Multi-Output Learning

This directory contains the official implementation of:

> **ADSI-MIMO: Adaptive Stain Imputation with Multi-Input and Multi-Output Learning for Multiplex Immunofluorescence Imaging**
> Xingnan Li, Priyanka Rana, Tuba N Gide, Nurudeen A Adegoke, Yizhe Mao, Shlomo Berkovsky, Enrico Coiera, James S Wilmott, Sidong Liu
> *bioRxiv*, 2025. [[Preprint]](https://doi.org/10.1101/2025.06.27.661891)

---

## Overview

Multiplex immunofluorescence (mIF) imaging captures simultaneous expression of multiple biomarkers across tissue sections, but the high cost and technical complexity of staining limits panel size in practice. **AdSI-MIMO** addresses this by imputing all missing stain channels simultaneously from any available subset of input markers — without requiring retraining for each input–output configuration.

Compared to the prior single-output method [SIMIF](../SIMIF/README.md), AdSI-MIMO brings three core advances:

| | SIMIF | AdSI-MIMO |
|---|---|---|
| Architecture | WGAN-GP, one model per target | MultiMAE encoder + multi-decoder |
| Output | Single stain per model | All non-fixed stains in one pass |
| Input flexibility | Fixed input set | Arbitrary subsets via masking |
| Masking strategy | Simple random masking | Adaptive Progressive Masking (APM) |

### Key Results

On melanoma (257 patients) and urothelial carcinoma (55 samples) datasets:
- **+18.4%** Pearson correlation for CD8 imputation across diverse input configurations
- **+48.1%** Pearson correlation for PD-L1 imputation on the external test set
- Strong generalization across unseen input combinations without retraining

---

## Architecture

The AdSI-MIMO framework consists of three components:

### 1 · Adaptive Progressive Masking (APM)

Biomarkers are split into two groups:

| Group | Examples | Masking |
|-------|----------|---------|
| **Base stains** (always available) | DAPI, autofluorescence | Fixed 40 % rate |
| **Antibody stains** (potential targets) | CD8, PD-L1, CD68, … | Progressive: 30 % → 75 % |

The antibody masking rate starts at **30 %** at epoch 0 and increases by **5 % every 25 epochs**, reaching a maximum of **75 %** at epoch 225. Tokens to mask within each domain are sampled via a **Dirichlet distribution** to produce diverse masking patterns across training.

Each 224 × 224 patch is divided into **16 × 16 pixel** tiles (196 patches per image). Unmasked patches are projected to **768 dimensions** (ViT-B standard) before being passed to the encoder.

### 2 · Multi-Head Transformer Encoder (ViT-B)

Unmasked patches from all input domains are concatenated and processed through a Vision Transformer encoder with multi-head self-attention layers. The encoder produces a unified latent representation **f_LF ∈ ℝ^(M × 768)** that captures cross-biomarker relationships across all available input channels.

### 3 · Cross–Self-Attention Decoder (per output domain)

Each output biomarker has its own independent decoder composed of:
1. **Cross-attention** — queries from the decoder attend to encoder features (K, V) as conditional context
2. **MLP** — dimension alignment
3. **Two self-attention blocks** — patch-level reconstruction

This design allows each decoder to exploit inter-biomarker dependencies learned by the shared encoder while generating spatially accurate predictions for its target stain.

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{L1}} + \lambda \cdot \mathcal{L}_{\text{MS-SSIM}}$$

Per-domain losses are automatically balanced by learnable **uncertainty weights** (Kendall et al., NeurIPS 2018): each domain's log-variance is a trainable parameter that controls its contribution to the total loss.

---

## Datasets

| Dataset | WSIs | Channels | Biomarkers | Patch size |
|---------|------|----------|------------|------------|
| Melanoma | 257 | 7 | CD68, SOX10, CD16, CD8, PD-L1, DAPI + autofluorescence | 224 × 224 |
| Urothelial carcinoma | 55 | 12 | CD3, CD4, CD8, CD68, SOX10, PD-L1, LAG3, Ki67, PD1, DAPI + autofluorescence | 224 × 224 |

**Preprocessing**: per-channel min-max normalization; background removal via Otsu's thresholding guided by DAPI; 224 × 224 patches extracted with 112-pixel overlap; WSIs capped at 2 500 patches.

**Data split** (WSI-level, to prevent data leakage):

| Split | Melanoma | Urothelial |
|-------|----------|------------|
| Train | 60 % (n = 126) | 60 % (n = 25) |
| Validation | 20 % (n = 43) | 20 % (n = 8) |
| Internal test | 20 % (n = 42) | 20 % (n = 8) |
| External test | separate cohort (n = 46) | separate cohort (n = 14) |

---

## Requirements

```
Python >= 3.8
torch >= 1.12
torchvision
einops
pytorch_msssim
numpy
pandas
scipy
scikit-image
tqdm
```

```bash
pip install torch torchvision einops pytorch_msssim numpy pandas scipy scikit-image tqdm
```

---

## Data Preparation

### Image files

Each image patch must be saved as a `.npy` file with shape `(C, H, W)`, where `C` is the total number of markers in the full panel. Channel order must match the marker panel definition.

### Marker panel file (`output.txt`)

A JSON-formatted text file mapping integer keys to marker names. Each value is a string of the form `<marker_name> <optional_extra>` — only the first space-separated token is used. Example:

```json
{"0": "dapi", "1": "autofluorescence", "2": "cd8", "3": "pd-l1", "4": "cd68", "5": "cd16", "6": "sox10"}
```

### Data split CSV

A CSV file with at least the following two columns:

| Column | Description |
|--------|-------------|
| `Image_Paths` | Absolute or relative path to each `.npy` patch file |
| `Split_Name` | One of `train`, `valid`, or `test` |

---

## Project Structure

```
AdSI-MIMO/
├── train_AdSIMIMO.py       # TrainerMMAE: MultiMAE training & validation loops
├── trainer_AdSIMIMO.py     # Base Trainer: data setup, checkpointing, evaluation pipeline
├── evaluate_AdSIMIMO.py    # Standalone evaluation script
├── dataloader_ssim.py      # MxIFReader: dataset class and data loaders
└── AdSIMIMO/               # Model package (adapted from MultiMAE)
    ├── __init__.py
    ├── multimae1.py         # Training-time MultiMAE (Dirichlet + curriculum masking)
    ├── multimae_e.py        # Evaluation-time MultiMAE (selective deterministic masking)
    ├── multimae_utils.py    # Transformer blocks, positional encodings
    ├── input_adapters.py    # PatchedInputAdapter for per-domain tokenisation
    ├── output_adapters.py   # SpatialOutputAdapter for spatial reconstruction
    ├── output_adapter_utils.py
    └── criterion.py         # MaskedL1Loss, MaskedMSELoss, MaskedCrossEntropyLoss
```

---

## Training

Edit the `__main__` block in `train_AdSIMIMO.py` and run:

```bash
cd AdSI-MIMO
python train_AdSIMIMO.py
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `marker_panel` | from `output.txt` | Full ordered list of marker names |
| `fixed_stain` | `["dapi", "autofluorescence"]` | Markers always kept as encoder input |
| `lr` | `1e-4` | Initial learning rate (Adam optimizer) |
| `percent` | `20` | % of training patches used per epoch |
| `img_size` | `224` | Spatial resolution (pixels) |
| `batch_size` | `16` | Batch size |
| `max_epochs` | `400` | Maximum training epochs |
| `patience` | `5` | Early-stopping patience (epochs) |
| `load_model_ckpt` | `False` | Resume from an existing checkpoint |
| `checkpoint_name` | — | Checkpoint filename inside `results_dir` |

Checkpoints (`checkpoint_{epoch}.pt`) bundle model weights, optimizer state, loss-balancer state, and epoch index. Training statistics (per-epoch loss, per-domain SSIM and Pearson-r) are written to `training_stats.csv`.

**Training configuration used in the paper**: 250 epochs, lr = 1e-4, Adam, batch size 16, Tesla V100 GPU (~4 days).

---

## Evaluation

Edit the `__main__` block in `evaluate_AdSIMIMO.py` and run:

```bash
cd AdSI-MIMO
python evaluate_AdSIMIMO.py
```

The `mask_biomarker` list in `evaluate_AdSIMIMO.py` controls which stains are treated as absent at inference time. Any output-domain stain **not** in `mask_biomarker` remains visible to the encoder as context — this is the *adaptive* aspect of AdSI-MIMO. Only the imputed stains are saved and scored.

For each imputed stain the script saves under `results_dir/{split}_{img_size}_{img_size}/`:
- **`{stain}/{image_name}.npy`** — array of shape `(2, H, W)` containing `[real, generated]`
- **`{stain}_stats.csv`** — per-image MAE, MSE, RMSE, PSNR, SSIM, Pearson-r, p-value

---

## Relationship to MultiMAE

The `AdSIMIMO/` package is adapted from [EPFL-VILAB/MultiMAE](https://github.com/EPFL-VILAB/MultiMAE). Key modifications for MxIF data:

1. All domains use **single-channel** (`num_channels=1`) input/output adapters.
2. A `real_output_index` argument controls which output-domain tokens are exposed to the encoder at test time (enables zero-shot multi-stain imputation).
3. The Adaptive Progressive Masking schedule replaces MultiMAE's uniform random masking.
4. Domain configuration is built **dynamically** from the marker panel — no hardcoded stain names.

If you use the MultiMAE components, please also cite:

```bibtex
@article{bachmann2022multimae,
  title   = {MultiMAE: Multi-modal Multi-task Masked Autoencoders},
  author  = {Bachmann, Roman and Mizrahi, David and Atanov, Andrei and Zamir, Amir},
  journal = {arXiv preprint arXiv:2204.01678},
  year    = {2022}
}
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{li2025adsimimo,
  title   = {ADSI-MIMO: Adaptive Stain Imputation with Multi-Input and Multi-Output
             Learning for Multiplex Immunofluorescence Imaging},
  author  = {Li, Xingnan and Rana, Priyanka and Gide, Tuba N and Adegoke, Nurudeen A
             and Mao, Yizhe and Berkovsky, Shlomo and Coiera, Enrico
             and Wilmott, James S and Liu, Sidong},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.27.661891},
  url     = {https://doi.org/10.1101/2025.06.27.661891}
}
```
