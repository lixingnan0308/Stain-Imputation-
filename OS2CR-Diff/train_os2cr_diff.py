"""
train_os2cr_diff.py — Training script for OS2CR-Diff.

Paper: "OS2CR-Diff: A Self-Refining Diffusion Framework for CD8 Imputation
        from One-Step Inference to Conditional Representation"
Li et al., IEEE BIBM 2025.  https://ieeexplore.ieee.org/document/11356570/

OS2CR-Diff is a v-prediction diffusion model that refines a coarse CD8 stain
estimate (produced by MAS in stage 1) into a high-fidelity imputation for
multiplex immunofluorescence (MxIF) imaging.

Input channel layout of each .npy patch file (8 channels total):
    0 : dapi             — fixed input  (original image)
    1 : cd68             — conditional input (MAS-generated)
    2 : cd8              — prior input (MAS-generated coarse CD8 estimate)
    3 : cd16             — conditional input (MAS-generated)
    4 : pd-l1            — conditional input (MAS-generated)
    5 : sox10            — conditional input (MAS-generated)
    6 : autofluorescence — fixed input  (original image)
    7 : real cd8         — reconstruction target (appended last by the dataloader)

The model receives three condition streams:
    condition1 (prior_cd8)        : channel 2  — 1-channel MAS-generated CD8
    condition2 (fixed_inputs)     : channels 0+6 — 2-channel DAPI + autofluorescence
    condition3 (conditional_inputs): channels 1+3+5 — 3-channel CD68/CD16/SOX10
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from tqdm import tqdm

from dataloader_ssim import MxIFReader
from network.unet import OS2CRDiff
from diffusion.noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def read_json_from_txt(file_path):
    """
    Reads a JSON-formatted text file and returns an ordered list of marker names.

    The expected format is a JSON object whose values are strings of the form
    'marker_name ...' (only the first space-separated token is used).

    Args:
        file_path (str): Path to the JSON-formatted .txt file.

    Returns:
        list[str]: Ordered list of marker names.
    """
    with open(file_path, 'r') as f:
        original_dict = json.loads(f.read())
    return [v.split(' ')[0] for v in original_dict.values()]


def set_seed(seed):
    """Sets the global random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def color_consistency_loss(pred, target):
    """
    Computes a color consistency loss between predicted and target images.

    Combines global per-channel mean/std statistics loss with a multi-scale
    average-pooling loss to encourage consistent intensity distributions.

    Args:
        pred   (torch.Tensor): Predicted image tensor of shape (B, C, H, W).
        target (torch.Tensor): Ground-truth image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    mean_loss = F.mse_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))
    std_loss  = F.mse_loss(pred.std(dim=[2, 3]),  target.std(dim=[2, 3]))
    local_loss = sum(
        F.mse_loss(F.avg_pool2d(pred, s), F.avg_pool2d(target, s))
        for s in [4, 8, 16]
    )
    return mean_loss + std_loss + 0.1 * local_loss


def train(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    set_seed(3817)

    diffusion_config = config['diffusion_params']
    dataset_config   = config['dataset_params']
    model_config     = config['model_params']
    train_config     = config['train_params']

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    stain_panel = read_json_from_txt(dataset_config['output_txt_path'])

    # Resolve channel roles from the config.  All marker names must appear in stain_panel.
    fixed_stain       = dataset_config['fixed_stain']        # e.g. ['dapi', 'autofluorescence']
    prior_stain       = dataset_config['prior_stain']        # e.g. 'cd8'
    conditional_stain = dataset_config['conditional_stain']  # e.g. ['cd68', 'cd16', 'sox10']

    fixed_idx       = [stain_panel.index(s) for s in fixed_stain]
    prior_idx       = stain_panel.index(prior_stain)
    conditional_idx = [stain_panel.index(s) for s in conditional_stain]

    print(f'Fixed      channels: {list(zip(fixed_stain, fixed_idx))}')
    print(f'Prior      channel : {prior_stain} → {prior_idx}')
    print(f'Conditional channels: {list(zip(conditional_stain, conditional_idx))}')

    train_dataset = MxIFReader(
        data_csv_path=dataset_config['csv_path'],
        split_name='train',
        marker_panel=stain_panel,
        input_markers=stain_panel,
        training=True,
        img_size=train_config['image_size'],
        percent=train_config['num_samples'],
    )
    train_loader = MxIFReader.get_data_loader(
        train_dataset,
        batch_size=train_config['batch_size'],
        training=True,
        num_workers=train_config['num_workers'],
    )

    model = OS2CRDiff(model_config).to(device)
    model.train()

    os.makedirs(train_config['task_name'], exist_ok=True)

    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print('Resuming from existing checkpoint.')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    num_epochs     = train_config['num_epochs']
    optimizer      = Adam(model.parameters(), lr=train_config['lr'])
    lambda_v       = 50.0   # v-prediction loss scale
    use_snr_weight = True   # SNR-weighted loss scaling

    for epoch_idx in range(num_epochs):
        losses = []

        for batch_idx, (input_batch, _, _) in enumerate(train_loader):
            optimizer.zero_grad()

            prior_cd8 = input_batch[:, prior_idx:prior_idx + 1, :, :].to(device, torch.float32)
            # Last channel is always the real CD8 reconstruction target.
            real_cd8  = input_batch[:, -1:, :, :].to(device, torch.float32)
            fixed_inputs = torch.cat(
                [input_batch[:, i:i + 1, :, :] for i in fixed_idx], dim=1
            ).to(device, torch.float32)
            conditional_inputs = torch.cat(
                [input_batch[:, i:i + 1, :, :] for i in conditional_idx], dim=1
            ).to(device, torch.float32)

            noise = torch.randn_like(real_cd8)
            t = torch.randint(
                0, diffusion_config['num_timesteps'],
                (real_cd8.shape[0],), device=device,
            )

            noisy_im = scheduler.add_noise(real_cd8, noise, t)
            v_target = scheduler.get_v_target(real_cd8, noise, t)

            v_pred = model(noisy_im, t, prior_cd8, fixed_inputs, conditional_inputs)

            loss_v = F.mse_loss(v_pred, v_target)

            # Recover x_0 for auxiliary color consistency diagnostics.
            x0_pred, _ = scheduler.predict_x0_and_noise_from_v(noisy_im, v_pred, t)
            loss_color  = color_consistency_loss(x0_pred, real_cd8)

            loss = lambda_v * loss_v

            if use_snr_weight:
                snr_weight = scheduler.get_snr_weight(t).mean()
                loss = loss * snr_weight

            if batch_idx % 50 == 0:
                print(
                    f'Epoch {epoch_idx + 1} | Batch {batch_idx + 1} '
                    f'| loss={loss.item():.4f} '
                    f'| v-loss={loss_v.item():.4f} '
                    f'| color={loss_color.item():.4f}'
                )

            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f'Finished epoch {epoch_idx + 1} | mean loss={np.mean(losses):.4f}')

        if (epoch_idx + 1) % 2 == 0:
            ckpt_stem = os.path.splitext(train_config['ckpt_name'])[0]
            ckpt_name = f'{ckpt_stem}_v_epoch_{epoch_idx + 1}.pth'
            torch.save(
                model.state_dict(),
                os.path.join(train_config['task_name'], ckpt_name),
            )

    print('Done training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OS2CR-Diff v-prediction training')
    parser.add_argument('--config', dest='config_path',
                        default='config/srs_conf.yaml', type=str)
    args = parser.parse_args()
    train(args)
