"""
sample_os2cr_diff.py — Inference script for OS2CR-Diff.

Paper: "OS2CR-Diff: A Self-Refining Diffusion Framework for CD8 Imputation
        from One-Step Inference to Conditional Representation"
Li et al., IEEE BIBM 2025.  https://ieeexplore.ieee.org/document/11356570/

Loads a trained OS2CR-Diff checkpoint and runs v-prediction DDPM or DDIM
sampling to impute the CD8 channel.  For every image the script saves:
  • A .npy file of shape (3, H, W): [real_cd8, prior_cd8, generated_cd8]
  • Per-image pixel-level metrics (MAE, MSE, RMSE, PSNR, SSIM, Pearson r)
    for both the generated result and the MAS prior, exported as CSV.
  • Visualisation grids saved as JPEG for a configurable batch index.

See train_os2cr_diff.py for the input channel layout.
"""

import argparse
import json
import os
import yaml

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats as st
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from dataloader_ssim import MxIFReader
from network.unet import OS2CRDiff
from diffusion.noise_scheduler import LinearNoiseScheduler
from train_os2cr_diff import read_json_from_txt

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def pixel_metrics(real, generated, max_val=255, baseline=False):
    """
    Computes pixel-level evaluation metrics between ground-truth and generated images.

    Args:
        real      (np.ndarray): Ground-truth image array.
        generated (np.ndarray): Generated image array.
        max_val   (float):      Maximum pixel value used for PSNR. Defaults to 255.
        baseline  (bool):       If True, skips Pearson correlation. Defaults to False.

    Returns:
        dict: MAE, MSE, RMSE, PSNR, SSIM, and (unless baseline) Corr and p-value.
    """
    real      = np.squeeze(real)
    generated = np.squeeze(generated)
    stats = {
        'MAE':  np.mean(np.abs(real - generated)),
        'MSE':  np.mean((real - generated) ** 2),
    }
    stats['RMSE'] = np.sqrt(stats['MSE'])
    stats['PSNR'] = 20 * np.log10(max_val) - 10.0 * np.log10(stats['MSE'] + 1e-8)
    stats['SSIM'] = ssim(real, generated, data_range=max_val, channel_axis=0)
    if not baseline:
        corr, p_value = st.pearsonr(real.flatten(), generated.flatten())
        stats['Corr']    = corr
        stats['p-value'] = p_value
    return stats


def sample_v_prediction(model, scheduler, diffusion_config,
                         val_loader, results_path,
                         prior_idx, fixed_idx, conditional_idx,
                         vis_batch_idx=0, use_ddim=False):
    """
    Runs v-prediction DDPM/DDIM sampling over the given data loader.

    Args:
        model            (nn.Module):             Trained OS2CRDiff.
        scheduler        (LinearNoiseScheduler):  Noise scheduler.
        diffusion_config (dict):                  Diffusion hyper-parameters.
        val_loader       (DataLoader):            Evaluation data loader.
        results_path     (str):                   Directory for .npy outputs.
        prior_idx        (int):                   Channel index of the MAS-generated CD8 prior.
        fixed_idx        (list[int]):             Channel indices of fixed inputs (DAPI, AF).
        conditional_idx  (list[int]):             Channel indices of conditional biomarkers.
        vis_batch_idx    (int):                   Batch index for JPEG visualisation. Defaults to 0.
        use_ddim         (bool):                  Use DDIM (~10× speedup). Defaults to False.
    """
    os.makedirs(results_path, exist_ok=True)
    vis_path = results_path + '_vis'

    stats_generated = {
        k: [] for k in ['Image_Name', 'Stain', 'MAE', 'MSE', 'SSIM', 'PSNR',
                         'RMSE', 'Corr', 'p-value']
    }
    stats_prior = {
        k: [] for k in ['Image_Name', 'Stain', 'MAE', 'MSE', 'SSIM', 'PSNR',
                         'RMSE', 'Corr', 'p-value']
    }

    for batch_idx, (input_batch, image_name_batch, _) in enumerate(tqdm(val_loader)):
        prior_cd8 = input_batch[:, prior_idx:prior_idx + 1, :, :].to(device, torch.float32)
        real_cd8  = input_batch[:, -1:, :, :].to(device, torch.float32)
        fixed_inputs = torch.cat(
            [input_batch[:, i:i + 1, :, :] for i in fixed_idx], dim=1
        ).to(device, torch.float32)
        conditional_inputs = torch.cat(
            [input_batch[:, i:i + 1, :, :] for i in conditional_idx], dim=1
        ).to(device, torch.float32)

        cd8_fake = torch.randn_like(real_cd8)

        if use_ddim:
            # DDIM: sample every 10th timestep for ~10× speedup.
            timesteps = list(range(0, diffusion_config['num_timesteps'], 10))[::-1]
        else:
            timesteps = reversed(range(diffusion_config['num_timesteps']))

        for i in tqdm(timesteps, leave=False):
            t = torch.full((cd8_fake.shape[0],), i, dtype=torch.long, device=device)
            v_pred = model(cd8_fake, t, prior_cd8, fixed_inputs, conditional_inputs)

            if use_ddim:
                cd8_fake, x0_pred = scheduler.sample_prev_timestep_v_ddim(
                    cd8_fake, v_pred, t, prev_t=t - 10
                )
            else:
                cd8_fake, x0_pred = scheduler.sample_prev_timestep_v(
                    cd8_fake, v_pred, t
                )

        # Rescale from [-1, 1] to [0, 1].
        for i, img_path in enumerate(image_name_batch):
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            real_np  = (real_cd8[i].detach().cpu().numpy()  + 1) / 2
            fake_np  = (torch.clamp(cd8_fake[i], -1., 1.).detach().cpu().numpy() + 1) / 2
            prior_np = (prior_cd8[i].detach().cpu().numpy() + 1) / 2

            # Save [real, prior, generated] array for downstream analysis.
            np.save(
                os.path.join(results_path, f'{image_name}.npy'),
                np.concatenate([real_np, prior_np, fake_np], axis=0),
            )

            # Compute metrics (scale to [0, 255]).
            m  = pixel_metrics(real_np * 255.0, fake_np  * 255.0, max_val=255)
            mp = pixel_metrics(real_np * 255.0, prior_np * 255.0, max_val=255)

            stats_generated['Image_Name'].append(image_name)
            stats_generated['Stain'].append('cd8')
            stats_prior['Image_Name'].append(image_name)
            stats_prior['Stain'].append('cd8_prior')
            for key in m:
                stats_generated[key].append(m[key])
            for key in mp:
                stats_prior[key].append(mp[key])

            # Visualisation for the chosen batch.
            if batch_idx == vis_batch_idx:
                os.makedirs(vis_path, exist_ok=True)
                _, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
                axs[0].imshow(prior_np.transpose(1, 2, 0), cmap='hot')
                axs[0].set_title('MAS prior (CD8)')
                axs[0].axis('off')
                axs[1].imshow(real_np.transpose(1, 2, 0), cmap='hot')
                axs[1].set_title('Real CD8')
                axs[1].axis('off')
                axs[2].imshow(fake_np.transpose(1, 2, 0), cmap='hot')
                axs[2].set_title('Generated CD8 (OS2CR-Diff)')
                axs[2].axis('off')
                plt.savefig(os.path.join(vis_path, f'{image_name}.jpg'),
                            format='jpg', dpi=300, bbox_inches='tight')
                plt.close()

        print(f'Finished sampling batch {batch_idx}')

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(stats_generated).to_csv(
        os.path.join('results', 'cd8_generated_stats.csv'), index=False
    )
    pd.DataFrame(stats_prior).to_csv(
        os.path.join('results', 'cd8_prior_stats.csv'), index=False
    )


def infer(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config     = config['model_params']
    train_config     = config['train_params']
    dataset_config   = config['dataset_params']

    stain_panel = read_json_from_txt(dataset_config['output_txt_path'])

    fixed_stain       = dataset_config['fixed_stain']
    prior_stain       = dataset_config['prior_stain']
    conditional_stain = dataset_config['conditional_stain']

    fixed_idx       = [stain_panel.index(s) for s in fixed_stain]
    prior_idx       = stain_panel.index(prior_stain)
    conditional_idx = [stain_panel.index(s) for s in conditional_stain]

    val_dataset = MxIFReader(
        data_csv_path=dataset_config['csv_path'],
        split_name='test',
        marker_panel=stain_panel,
        input_markers=stain_panel,
        training=False,
        img_size=train_config['image_size'],
        percent=train_config['num_samples'],
    )
    val_loader = MxIFReader.get_data_loader(
        val_dataset,
        batch_size=train_config['batch_size'],
        training=False,
        num_workers=train_config['num_workers'],
    )

    model = OS2CRDiff(model_config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(train_config['task_name'], train_config['ckpt_load_name']),
            map_location=device,
        )
    )
    model.eval()

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    ckpt_stem    = os.path.splitext(train_config['ckpt_load_name'])[0]
    results_path = os.path.join('results', ckpt_stem)

    with torch.no_grad():
        sample_v_prediction(
            model, scheduler, diffusion_config,
            val_loader, results_path,
            prior_idx=prior_idx,
            fixed_idx=fixed_idx,
            conditional_idx=conditional_idx,
            vis_batch_idx=args.vis_batch_idx,
            use_ddim=args.use_ddim,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OS2CR-Diff v-prediction sampling')
    parser.add_argument('--config', dest='config_path',
                        default='config/srs_conf_evaluate_self_v2.yaml', type=str)
    parser.add_argument('--use_ddim', action='store_true', default=True,
                        help='Use DDIM sampling (~10× faster than DDPM).')
    parser.add_argument('--vis_batch_idx', type=int, default=0,
                        help='Batch index for JPEG visualisation output.')
    args = parser.parse_args()
    infer(args)
