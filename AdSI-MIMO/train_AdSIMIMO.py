"""
train_AdSIMIMO.py — AdSI-MIMO training script.

Defines TrainerMMAE, which extends the base Trainer with a MultiMAE-based
architecture adapted for multiplexed immunofluorescence (MxIF) stain imputation.

The MultiMAE backbone is adapted from:
    MultiMAE: Multi-modal Multi-task Masked Autoencoders
    https://github.com/EPFL-VILAB/MultiMAE

Each fluorescence marker is treated as an independent "domain" with its own
patch-level input adapter and spatial output adapter. During training, a
progressively increasing fraction of output-domain tokens is masked, forcing
the model to impute them from the always-visible fixed channels (e.g., DAPI).
An uncertainty-weighted loss (Kendall et al., NeurIPS 2018) balances the
per-domain L1 and MS-SSIM objectives.
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ms_ssim
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from trainer_AdSIMIMO import Trainer, read_json_from_txt
from AdSIMIMO.multimae_utils import Block, trunc_normal_
from AdSIMIMO.input_adapters import PatchedInputAdapter
from AdSIMIMO.output_adapters import SpatialOutputAdapter
from AdSIMIMO.criterion import MaskedL1Loss
from AdSIMIMO.multimae1 import pretrain_multimae_base
from AdSIMIMO.multimae_e import pretrain_multimae_base as pretrain_multimae_base_e

class TrainerMMAE(Trainer):
    def __init__(self, marker_panel, fixed_stain, results_dir, lr=0.002, seed=1):
        """
        Trainer for the AdSI-MIMO MultiMAE-based stain imputation model.

        Args:
            marker_panel (list): Ordered list of marker names matching the channel order
                                 in the .npy image files.
            fixed_stain (list): Markers that are always available as input and are never
                                imputed (e.g., ['dapi', 'autofluorescence']).
            results_dir (str): Directory where checkpoints and results are saved.
            lr (float): Initial learning rate. Defaults to 0.002.
            seed (int): Global random seed for reproducibility. Defaults to 1.
        """
        super().__init__(marker_panel, fixed_stain, results_dir, lr)
        self.seed = seed
        self.mask = True

    def init_model(self, input_domains, output_domains,
                   patch_size=16, decoder_dim=256, channel=1, stride_level=1,
                   depth=2, num_heads=8, use_task_queries=True, use_xattn=True):
        """
        Builds the MultiMAE model with per-domain input and output adapters.

        Each marker in the panel gets a PatchedInputAdapter that splits its
        single-channel image into non-overlapping patches. Non-fixed markers
        additionally get a SpatialOutputAdapter that reconstructs full-resolution
        feature maps via a lightweight cross-attention decoder.

        Args:
            input_domains (list): Marker names provided as encoder input.
            output_domains (list): Marker names to be reconstructed by the decoder.
            patch_size (int): Patch size for the ViT tokenizer. Defaults to 16.
            decoder_dim (int): Token dimensionality for output adapters. Defaults to 256.
            channel (int): Number of channels per domain (always 1 for MxIF). Defaults to 1.
            stride_level (int): Spatial stride scaling relative to the ViT grid. Defaults to 1.
            depth (int): Number of transformer layers in each output adapter. Defaults to 2.
            num_heads (int): Attention heads in each output adapter. Defaults to 8.
            use_task_queries (bool): If True, learnable task queries are prepended. Defaults to True.
            use_xattn (bool): If True, cross-attention to encoder tokens is used. Defaults to True.

        Returns:
            tuple:
                model (nn.Module): Initialized MultiMAE model.
                DOMAIN_CONF (dict): Per-domain configuration dict (passed to init_loss_function).
        """
        # Build domain configuration dynamically from the marker panel.
        # Fixed stains (e.g., dapi, autofluorescence) only have an input adapter;
        # all other markers also have an output adapter and a reconstruction loss.
        DOMAIN_CONF = {}
        for domain in self.marker_panel:
            entry = {
                'channels': channel,
                'stride_level': stride_level,
                'input_adapter': partial(PatchedInputAdapter, num_channels=1),
            }
            if domain not in self.fixed_stain:
                entry['output_adapter'] = partial(SpatialOutputAdapter, num_channels=1)
                entry['loss'] = MaskedL1Loss
            DOMAIN_CONF[domain] = entry

        input_adapters = {
            domain: DOMAIN_CONF[domain]['input_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=patch_size,
            )
            for domain in input_domains
        }
        output_adapters = {
            domain: DOMAIN_CONF[domain]['output_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=patch_size,
                dim_tokens=decoder_dim,
                depth=depth,
                num_heads=num_heads,
                use_task_queries=use_task_queries,
                task=domain,
                context_tasks=list(output_domains),
                use_xattn=use_xattn,
            )
            for domain in output_domains
        }

        model = pretrain_multimae_base(
            input_adapters, output_adapters,
            input_domains=self.input_domains,
            output_domains=self.output_domains,
        )
        return model, DOMAIN_CONF
    
    def init_model_eval(self, input_domains, output_domains,
                   patch_size=16, decoder_dim=256, channel=1, stride_level=1,
                   depth=2, num_heads=8, use_task_queries=True, use_xattn=True):
        """
        Builds the MultiMAE model with per-domain input and output adapters.

        Each marker in the panel gets a PatchedInputAdapter that splits its
        single-channel image into non-overlapping patches. Non-fixed markers
        additionally get a SpatialOutputAdapter that reconstructs full-resolution
        feature maps via a lightweight cross-attention decoder.

        Args:
            input_domains (list): Marker names provided as encoder input.
            output_domains (list): Marker names to be reconstructed by the decoder.
            patch_size (int): Patch size for the ViT tokenizer. Defaults to 16.
            decoder_dim (int): Token dimensionality for output adapters. Defaults to 256.
            channel (int): Number of channels per domain (always 1 for MxIF). Defaults to 1.
            stride_level (int): Spatial stride scaling relative to the ViT grid. Defaults to 1.
            depth (int): Number of transformer layers in each output adapter. Defaults to 2.
            num_heads (int): Attention heads in each output adapter. Defaults to 8.
            use_task_queries (bool): If True, learnable task queries are prepended. Defaults to True.
            use_xattn (bool): If True, cross-attention to encoder tokens is used. Defaults to True.

        Returns:
            tuple:
                model (nn.Module): Initialized MultiMAE model.
                DOMAIN_CONF (dict): Per-domain configuration dict (passed to init_loss_function).
        """
        # Build domain configuration dynamically from the marker panel.
        # Fixed stains (e.g., dapi, autofluorescence) only have an input adapter;
        # all other markers also have an output adapter and a reconstruction loss.
        DOMAIN_CONF = {}
        for domain in self.marker_panel:
            entry = {
                'channels': channel,
                'stride_level': stride_level,
                'input_adapter': partial(PatchedInputAdapter, num_channels=1),
            }
            if domain not in self.fixed_stain:
                entry['output_adapter'] = partial(SpatialOutputAdapter, num_channels=1)
                entry['loss'] = MaskedL1Loss
            DOMAIN_CONF[domain] = entry

        input_adapters = {
            domain: DOMAIN_CONF[domain]['input_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=patch_size,
            )
            for domain in input_domains
        }
        output_adapters = {
            domain: DOMAIN_CONF[domain]['output_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=patch_size,
                dim_tokens=decoder_dim,
                depth=depth,
                num_heads=num_heads,
                use_task_queries=use_task_queries,
                task=domain,
                context_tasks=list(output_domains),
                use_xattn=use_xattn,
            )
            for domain in output_domains
        }

        model = pretrain_multimae_base_e(
            input_adapters, output_adapters,
            input_domains=self.input_domains,
            output_domains=self.output_domains,
        )
        return model, DOMAIN_CONF

    def init_optimizer(self, model, filter_bias_and_bn=True, skip_list=None):
        """
        Creates an AdamW optimizer for the model and loss-balancer parameters.

        When model is a dict with 'model' and 'balancer' keys, both parameter
        groups are added with lr_scale=1. When model is a plain nn.Module, all
        trainable parameters are optimized directly.

        Args:
            model (nn.Module | dict): Either the main network or a dict with
                                      'model' and 'balancer' entries.
            filter_bias_and_bn (bool): Unused placeholder for future layer-wise
                                       decay support. Defaults to True.
            skip_list (list | None): Parameter names excluded from weight decay.

        Returns:
            torch.optim.AdamW: Configured optimizer.
        """
        weight_decay = 1e-5

        if isinstance(model, dict):
            # Optimise backbone and uncertainty-weighting balancer jointly
            parameters = [
                {
                    'params': [p for _, p in model['model'].named_parameters()
                               if p.requires_grad],
                    'lr_scale': 1.,
                },
                {
                    'params': [p for _, p in model['balancer'].named_parameters()
                               if p.requires_grad],
                    'lr_scale': 1.,
                },
            ]
        else:
            parameters = model.parameters()

        opt_args = dict(lr=self.lr, weight_decay=weight_decay)
        print('Optimizer settings:', opt_args)
        return optim.AdamW(parameters, **opt_args)

    def init_loss_function(self, output_domains, DOMAIN_CONF):
        """
        Instantiates per-domain reconstruction losses and the uncertainty-weighting balancer.

        The UncertaintyWeightingStrategy (Kendall et al., NeurIPS 2018) learns
        per-task log-variance parameters that automatically balance the contribution
        of each domain's loss. Tasks whose output was dropped (loss == 0) are excluded
        from the weighted sum to avoid a trivial solution.

        Args:
            output_domains (list): Marker names that are reconstructed.
            DOMAIN_CONF (dict): Per-domain configuration (must contain 'loss' and
                                'stride_level' for each domain in output_domains).

        Returns:
            tuple:
                loss_balancer (UncertaintyWeightingStrategy): Learnable task-weighting module.
                tasks_loss_fn (dict): Maps domain name to its masked pixel-level loss.
        """
        class UncertaintyWeightingStrategy(nn.Module):
            """
            Learns per-task log-variance weights to balance multi-task losses
            (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
            Losses for Scene Geometry and Semantics", NeurIPS 2018).
            """
            def __init__(self, tasks):
                super().__init__()
                self.tasks = tasks
                # log(sigma^2) for each task; initialised to 0 (sigma = 1)
                self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

            def forward(self, task_losses):
                losses_tensor = torch.stack(list(task_losses.values()))
                # Mask tasks with zero loss so they do not contribute via log_var
                non_zero_mask = (losses_tensor != 0.0)
                losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars
                losses_tensor = losses_tensor * non_zero_mask
                weighted = task_losses.copy()
                weighted.update(zip(weighted, losses_tensor))
                return weighted

        loss_balancer = UncertaintyWeightingStrategy(tasks=output_domains)
        tasks_loss_fn = {
            domain: DOMAIN_CONF[domain]['loss'](
                patch_size=16, stride=DOMAIN_CONF[domain]['stride_level']
            )
            for domain in output_domains
        }
        return loss_balancer, tasks_loss_fn

    def get_mask_percentage(self, current_epoch):
        """
        Returns the token-masking ratio for the current epoch using a step schedule.

        The masking ratio starts at 30% and increases by 5% every 20 epochs,
        capped at 80%. This curriculum encourages the model to first learn from
        lightly masked inputs before facing the harder highly-masked setting.

        Args:
            current_epoch (int): Zero-indexed training epoch.

        Returns:
            float: Masking ratio in [0, 1].
        """
        initial_pct = 30
        step_size = 5
        epochs_per_step = 20
        max_pct = 80
        steps = current_epoch // epochs_per_step
        return min(initial_pct + steps * step_size, max_pct) * 0.01

    def train_loop(self, data_loader, epoch, input_domains, output_domains):
        """
        Single-epoch training pass over the data loader.

        For each batch the full marker panel is split into per-domain tensors,
        a curriculum-determined fraction of output-domain tokens is masked
        (not visible to the encoder), and the combined L1 + MS-SSIM loss is
        back-propagated with gradient clipping (max norm = 1.0).

        Args:
            data_loader (DataLoader): Training data loader.
            epoch (int): Current epoch index (used for mask schedule).
            input_domains (list): Ordered marker names for encoder input channels.
            output_domains (list): Ordered marker names for decoder reconstruction targets.

        Returns:
            float: Mean total loss over all batches.
        """
        self.model = self.model.to(self.device)
        self.loss_balancer = self.loss_balancer.to(self.device)
        self.model.train()

        total_error = 0.0
        batch_count = len(data_loader)

        for batch_idx, (input_batch, _, _) in enumerate(data_loader):
            input_batch = input_batch.to(self.device)

            # Build per-domain dict of single-channel tensors (B, 1, H, W)
            mae_batch = {
                input_domains[i]: input_batch[:, i:i + 1, :, :]
                for i in range(len(input_domains))
            }

            # Sanity check: warn on NaN/Inf in any input domain
            for domain, batch in mae_batch.items():
                if torch.isnan(batch).any() or torch.isinf(batch).any():
                    print(f'Warning: NaN/Inf in input domain "{domain}"')

            # Compute curriculum masking ratio and number of visible output tokens
            masked_pct = self.get_mask_percentage(epoch)
            num_encoded_tokens = int(len(output_domains) * 196 * (1 - masked_pct))

            outputs, task_masks = self.model(x=mae_batch,
                                             num_encoded_tokens=num_encoded_tokens)

            for task, output in outputs.items():
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f'Warning: NaN/Inf in output for task "{task}"')

            # Per-task loss: masked L1 + (1 - MS-SSIM)
            task_losses = {}
            for task in outputs:
                target = mae_batch[task]
                loss_l1 = self.tasks_loss_fn[task](
                    outputs[task], target, mask=task_masks.get(task, None)
                )
                loss_ssim = ms_ssim(outputs[task], target, data_range=1.0,
                                    size_average=True)
                task_loss = loss_l1 + (1 - loss_ssim)
                if torch.isnan(task_loss).any() or torch.isinf(task_loss).any():
                    print(f'Warning: NaN/Inf loss for task "{task}"')
                task_losses[task] = task_loss

            weighted_task_losses = self.loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())
            loss_value = sum(task_losses.values()).item()

            self.optimizer.zero_grad()
            loss.backward()

            # Warn on exploding gradients before clipping
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).all() or torch.isinf(param.grad).all():
                        print(f'Warning: NaN/Inf gradient for "{name}"')

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            print(f'Train [{batch_idx}/{batch_count}]  loss={loss_value:.6f}', end='\r')
            total_error += loss_value

        print()
        return total_error / batch_count

    def valid_loop(self, data_loader, epoch, input_domains, output_domains):
        """
        Single-epoch validation pass with per-domain Pearson correlation and SSIM.

        Args:
            data_loader (DataLoader): Validation data loader.
            epoch (int): Current epoch index (used for mask schedule).
            input_domains (list): Ordered marker names for encoder input channels.
            output_domains (list): Ordered marker names for decoder reconstruction targets.

        Returns:
            tuple:
                float: Mean total loss over all batches.
                list[float]: Per-domain mean Pearson correlation.
                list[float]: Per-domain mean SSIM.
        """
        self.model = self.model.to(self.device)
        self.loss_balancer = self.loss_balancer.to(self.device)
        self.model.eval()

        total_error = 0.0
        batch_count = len(data_loader)
        real_images = []
        fake_images = []
        corr_list = {d: [] for d in output_domains}
        ssim_scores = {d: [] for d in output_domains}

        with torch.no_grad():
            for batch_idx, (input_batch, _, _) in enumerate(data_loader):
                input_batch = input_batch.to(self.device)

                mae_batch = {
                    input_domains[i]: input_batch[:, i:i + 1, :, :]
                    for i in range(len(input_domains))
                }

                masked_pct = self.get_mask_percentage(epoch)
                num_encoded_tokens = int(len(output_domains) * 196 * (1 - masked_pct))

                outputs, task_masks = self.model(x=mae_batch,
                                                 num_encoded_tokens=num_encoded_tokens)

                task_losses = {}
                for task in outputs:
                    target = mae_batch[task]
                    loss_l1 = self.tasks_loss_fn[task](
                        outputs[task], target, mask=task_masks.get(task, None)
                    )
                    loss_ssim = ms_ssim(outputs[task], target, data_range=1.0,
                                        size_average=True)
                    task_losses[task] = loss_l1 + (1 - loss_ssim)

                weighted_task_losses = self.loss_balancer(task_losses)
                loss_value = sum(task_losses.values()).item()

                # Accumulate predictions and targets for metric computation
                fake_images.append(
                    np.concatenate([
                        outputs[d].to(torch.float32).cpu().numpy()
                        for d in output_domains
                    ], axis=1)
                )
                real_images.append(
                    np.concatenate([
                        mae_batch[d].to(torch.float32).cpu().numpy()
                        for d in output_domains
                    ], axis=1)
                )

                print(f'Valid [{batch_idx}/{batch_count}]  loss={loss_value:.6f}', end='\r')
                total_error += loss_value

        print()

        real_images = np.concatenate(real_images, axis=0)
        fake_images = np.concatenate(fake_images, axis=0)

        # Compute per-domain, per-image Pearson correlation and SSIM
        for j, domain in enumerate(output_domains):
            for i in range(real_images.shape[0]):
                real_flat = real_images[i, j].flatten()
                fake_flat = fake_images[i, j].flatten()

                # Guard against constant images (correlation is undefined)
                if np.all(real_flat == real_flat[0]) or np.all(fake_flat == fake_flat[0]):
                    corr_list[domain].append(0.0)
                else:
                    corr, _ = pearsonr(real_flat, fake_flat)
                    corr_list[domain].append(corr)

                ssim_scores[domain].append(
                    ssim(real_images[i, j], fake_images[i, j],
                         channel_axis=None, data_range=1.0)
                )

        mean_corr = [np.mean(corr_list[d]) for d in output_domains]
        mean_ssim = [np.mean(ssim_scores[d]) for d in output_domains]

        print(f'Validation  Pearson={mean_corr}  SSIM={mean_ssim}')
        return total_error / batch_count, mean_corr, mean_ssim


if __name__ == '__main__':
    # ---- Marker panel -------------------------------------------------------
    stain_panel = read_json_from_txt('./output.txt')
    fixed_stain = ['dapi', 'autofluorescence']

    # ---- Paths --------------------------------------------------------------
    data_csv_path = './try_scale.csv'
    results_dir = './results_AdSIMIMO'

    # ---- Build trainer and start training -----------------------------------
    obj = TrainerMMAE(
        marker_panel=stain_panel,
        fixed_stain=fixed_stain,
        results_dir=results_dir,
        lr=0.0001,
        seed=1,
    )

    obj.train(
        data_csv_path,
        percent=20,
        img_size=224,
        batch_size=16,
        num_workers=4,
        max_epochs=400,
        minimum_epochs=380,
        patience=5,
        load_model_ckpt=False,
    )
