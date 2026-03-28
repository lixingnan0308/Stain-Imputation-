# SIMIF: Stain Imputation in Multiplex Immunofluorescence Imaging

This repository contains the official implementation of the paper:

> **Stain Imputation in Multiplex Immunofluorescence Imaging**
> Accepted at IEEE International Symposium on Biomedical Imaging (ISBI) 2025
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10981016/)

## Overview

SIMIF imputes missing stain channels in multiplexed immunofluorescence (MxIF) images using a Wasserstein GAN with Gradient Penalty (WGAN-GP). Given a subset of available stain channels (e.g., DAPI, CD4, CD8) as input, the model generates the missing target channel (e.g., PD-L1) at the image-patch level.

Key design choices:
- **Generator**: U-Net with skip connections (`init_features=128`)
- **Discriminator**: PatchGAN-style network with Instance Normalization
- **Loss**: WGAN-GP adversarial loss (λ=10) + L1 pixel reconstruction (λ=50)
- **Curriculum masking**: randomly zeros out non-fixed input channels during training to simulate partially missing panels, with masking probability increasing across epochs
- **Validation metric**: equal-weighted combination of Pearson correlation and SSIM (0.5 × Pearson + 0.5 × SSIM)

## Requirements

```
Python >= 3.8
torch >= 1.12
torchvision
numpy
pandas
scipy
scikit-image
captum
```

Install dependencies:

```bash
pip install torch torchvision numpy pandas scipy scikit-image captum
```

## Data Preparation

### Image files

Each image patch must be saved as a `.npy` file with shape `(C, H, W)`, where `C` is the total number of markers in the full panel. Channels must be in the same order as the marker panel definition.

### Marker panel file (`output.txt`)

A JSON-formatted text file mapping channel indices to marker names. Each value is a string of the form `<marker_name> <optional_extra>`. Only the first token is used as the marker name. Example:

```json
{"0": "dapi", "1": "autofluorescence", "2": "cd4", "3": "cd8", "4": "pd-l1"}
```

### Data split CSV

A CSV file with at least the following two columns:

| Column | Description |
|--------|-------------|
| `Image_Paths` | Absolute or relative path to each `.npy` patch file |
| `Split_Name` | One of `train`, `valid`, or `test` |

Example:

```
Image_Paths,Split_Name
/data/patches/slide1_patch001.npy,train
/data/patches/slide1_patch002.npy,valid
/data/patches/slide2_patch001.npy,test
```

## Project Structure

```
SIMIF/
├── train_simif.py       # TrainerCGAN: WGAN-GP training and validation loops
├── trainer_simif.py     # Base Trainer class: data setup, checkpointing, evaluation, attributions
├── dataloader.py        # MxIFReader: dataset class and data loaders
├── networks_base.py     # Generator (U-Net) and Discriminator architectures
└── evaluation_simif.py  # Standalone evaluation script
```

## Training

Edit the `__main__` block in `train_simif.py` to set your paths and hyperparameters, then run:

```bash
cd SIMIF
python train_simif.py
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `marker_panel` | Full list of marker names in channel order (loaded from `output.txt`) |
| `fixed_markers` | Markers always kept as input (e.g., `["dapi", "autofluorescence"]`) |
| `potential_output_markers` | Markers that may be used as imputation targets (e.g., `["cd8", "pd-l1"]`) |
| `target_marker` | The specific marker to impute in this run (e.g., `["cd8"]`) |
| `lr` | Learning rate (default: `0.0002`) |
| `percent` | Percentage of training data to use (default: `50`) |
| `img_size` | Spatial patch size in pixels (default: `224`) |
| `batch_size` | Batch size (default: `16`) |
| `max_epochs` | Maximum training epochs (default: `400`) |
| `patience` | Early stopping patience in epochs (default: `5`) |

Checkpoints are saved to `results_dir` as `checkpoint_{epoch}.pt` (generator) and `checkpoint_d_{epoch}.pt` (discriminator). Training statistics are written to `training_stats.csv` in the same directory.

## Evaluation

Edit the `__main__` block in `evaluation_simif.py` to point to your data and checkpoint, then run:

```bash
cd SIMIF
python evaluation_simif.py
```

Per-image metrics computed: MAE, MSE, RMSE, PSNR, SSIM, Pearson correlation.

Output files per evaluated stain:
- `.npy` files containing real and generated channels concatenated along the channel axis
- A `*_stats.csv` with per-image metrics

## Devices

The code automatically selects the compute device:
- **macOS**: Apple MPS (if available), otherwise CPU — uses `bfloat16` precision
- **Linux / Windows**: CUDA GPU (if available), otherwise CPU — uses `float32` precision

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{simif2025,
  title     = {Stain Imputation in Multiplex Immunofluorescence Imaging},
  booktitle = {2025 IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2025},
  url       = {https://ieeexplore.ieee.org/document/10981016/}
}
```
