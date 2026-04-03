# OS2CR-Diff

Code for the paper:

> **OS2CR-Diff: A Self-Refining Diffusion Framework for CD8 Imputation from One-Step Inference to Conditional Representation**
> Xingnan Li, Priyanka Rana, Tuba N Gide, Nurudeen A Adegoke, Yizhe Mao, James S Wilmott, Sidong Liu
> *2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, pp. 1062–1069
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11356570/)

---

## Overview

**OS2CR** stands for *One-Step to Conditional Representation* — the model takes a one-step MAS prior and refines it through a diffusion process conditioned on multi-channel representations of the tissue microenvironment.

OS2CR-Diff is a **two-stage** stain imputation framework for multiplex immunofluorescence (MxIF) imaging that targets the CD8 channel:

| Stage | Method | Output |
|-------|--------|--------|
| 1 | [MAS](https://github.com/ZixiaZ/MAS) | Coarse estimates of all antibody channels (CD68, CD8, CD16, PD-L1, SOX10) |
| 2 | OS2CR-Diff (this repo) | Refined, high-fidelity CD8 imputation conditioned on stage-1 outputs |

The diffusion model uses **v-prediction** with SNR-weighted loss and injects three condition streams into a conditional UNet:

| Stream | Config key | Description |
|--------|-----------|-------------|
| Prior CD8 | `prior_stain` | MAS-generated CD8 — coarse spatial prior |
| Fixed inputs | `fixed_stain` | DAPI + autofluorescence — always-available originals |
| Conditional biomarkers | `conditional_stain` | Other MAS-generated channels (e.g. CD68, CD16, SOX10) |

All channel role assignments are specified by **marker name** in the config YAML — no hardcoded indices in the code.

Condition features are fused at four UNet blocks (`down_3`, `mid1`, `mid2`, `up_0`) using a **shared multi-scale condition encoder** with cross-attention between streams and adaptive spatial-channel gating.

---

## Repository structure

```
OS2CR-Diff/
├── train_os2cr_diff.py   # Training entry point
├── sample_os2cr_diff.py  # Inference / evaluation entry point
├── dataloader_ssim.py    # MxIFReader dataset for .npy image patches
├── output.txt            # JSON marker panel (channel order)
├── configs/
│   ├── train.yaml        # Training configuration
│   └── eval.yaml         # Evaluation configuration
├── network/
│   └── unet.py           # OS2CRDiff model with SharedMultiScaleConditionEncoder
└── diffusion/
    └── noise_scheduler.py  # Linear DDPM scheduler with v-prediction support
```

---

## Input data format

### Image files

Each patch is a `.npy` file of shape `(C, H, W)`. The channel order must match `output.txt`. Channel roles (which markers are fixed, prior, or conditional) are declared in the config — see [Configuration](#configuration) below.

For the default melanoma panel:

| Index | Marker | Role |
|-------|--------|------|
| 0 | DAPI | Fixed input (original image) |
| 1 | CD68 | Conditional input (MAS-generated) |
| 2 | CD8 | Prior input (MAS-generated coarse estimate) |
| 3 | CD16 | Conditional input (MAS-generated) |
| 4 | PD-L1 | MAS-generated (not used by default) |
| 5 | SOX10 | Conditional input (MAS-generated) |
| 6 | Autofluorescence | Fixed input (original image) |
| 7 | **Real CD8** | Reconstruction target — appended as the last channel |

Values should be in **[0, 1]**; the dataloader normalises to [-1, 1] internally.

### Marker panel file (`output.txt`)

JSON mapping integer keys to marker names. Only the first space-separated token of each value is used:

```json
{"0": "dapi", "1": "cd68 (opal 520)", "2": "cd8 (opal 570)",
 "3": "cd16 (opal 620)", "4": "pd-l1 (opal 650)",
 "5": "sox10 (opal 690)", "6": "autofluorescence"}
```

### Split CSV

| Column | Description |
|--------|-------------|
| `Image_Paths` | Absolute path to the `.npy` patch file |
| `Split` | One of `train`, `valid`, `test` |

---

## Configuration

Both training and evaluation configs share the same structure. The `dataset_params` section controls data paths and **channel role assignments**:

```yaml
dataset_params:
  csv_path: 'internal.csv'
  output_txt_path: 'output.txt'
  fixed_stain:       ['dapi', 'autofluorescence']  # always-available originals
  prior_stain:       'cd8'                         # MAS-generated coarse CD8 prior
  conditional_stain: ['cd68', 'cd16', 'sox10']     # other MAS-generated biomarkers

diffusion_params:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

model_params:
  im_channels: 1
  down_channels: [64, 128, 256, 256, 512]
  mid_channels: [512, 512, 256]
  down_sample: [True, True, True, True]
  time_emb_dim: 128
  num_down_layers: 2
  num_mid_layers: 2
  num_up_layers: 2
  num_heads: 4
  use_attention: [False, False, True, True]
  use_condition: true
  condition_blocks: ["down_3", "mid1", "mid2", "up_0"]

train_params:
  task_name: 'mif_diff'
  image_size: 224
  batch_size: 8
  num_workers: 4
  num_epochs: 1000
  num_samples: 70       # % of training patches to use per epoch
  lr: 0.00005
  ckpt_name: 'ddpm_ckpt.pth'
```

To adapt to a different panel, update `output.txt` and the three `*_stain` keys — no Python changes required.

---

## Requirements

```
Python >= 3.9
torch >= 2.0
torchvision
pyyaml
numpy
pandas
scipy
scikit-image
pytorch_msssim
matplotlib
tqdm
```

---

## Training

```bash
cd OS2CR-Diff
python train_os2cr_diff.py --config configs/train.yaml
```

Checkpoints are saved every 2 epochs as `{ckpt_name}_v_epoch_{epoch}.pth` inside `task_name/`.

---

## Evaluation

Set `ckpt_load_name` in `configs/eval.yaml` to the checkpoint you want to evaluate, then:

```bash
python sample_os2cr_diff.py --config configs/eval.yaml [--use_ddim]
```

`--use_ddim` enables DDIM sampling (~10× faster than DDPM).

For each test image the script saves under `results/{ckpt_stem}/`:
- `{image_name}.npy` — shape `(3, H, W)` containing `[real_cd8, prior_cd8, generated_cd8]`
- `results/cd8_generated_stats.csv` — MAE, MSE, RMSE, PSNR, SSIM, Pearson r for the diffusion output
- `results/cd8_prior_stats.csv` — same metrics for the MAS stage-1 prior (baseline comparison)

---

## Stage 1: MAS prior generation

Before running OS2CR-Diff, generate the coarse stain estimates with MAS:

> **MAS** — Multi-stain Adaptive Synthesis
> [[GitHub]](https://github.com/ZixiaZ/MAS)

Concatenate the MAS-generated channels into a `.npy` file following the channel order in `output.txt`, then append the real CD8 channel last to produce the 8-channel files expected by this dataloader.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{11356570,
  author    = {Li, Xingnan and Rana, Priyanka and Gide, Tuba N and Adegoke, Nurudeen A
               and Mao, Yizhe and Wilmott, James S and Liu, Sidong},
  booktitle = {2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  title     = {OS2CR-Diff: A Self-Refining Diffusion Framework for CD8 Imputation
               from One-Step Inference to Conditional Representation},
  year      = {2025},
  pages     = {1062--1069},
  doi       = {10.1109/BIBM66473.2025.11356570},
  url       = {https://ieeexplore.ieee.org/document/11356570/}
}
```
