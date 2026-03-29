
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

"""
multimae_e.py — Evaluation variant of MultiMAE for AdSI-MIMO.

Unlike the training model (multimae1.py) which uses Dirichlet-sampled random
masking, this module applies deterministic selective masking at inference time:

- Domains listed in `real_output_index` (the stains to impute) are fully masked
  from the encoder — their channels are excluded from self-attention.
- All other domains (fixed channels such as DAPI, and any context output domains
  not being imputed) remain fully visible to the encoder.

This enables AdSI-MIMO's adaptive inference: the same model can impute any
subset of output stains while using the remaining stains as additional context,
without retraining.
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from einops import repeat
from torch import nn

from .multimae_utils import Block, trunc_normal_
from .input_adapters import PatchedInputAdapter
from .output_adapters import SpatialOutputAdapter


__all__ = ['pretrain_multimae_base', 'pretrain_multimae_large']


class MultiMAE(nn.Module):
    """MultiMAE backbone for evaluation with selective domain masking.

    At inference time, a caller-specified subset of output domains is masked
    from the encoder (simulating absent stains), while fixed domains and any
    remaining output domains act as visible context. Output adapters
    reconstruct the masked domains via cross-attention to the encoded tokens.

    Args:
        input_adapters  (dict): {domain: PatchedInputAdapter} tokenisers.
        output_adapters (dict): {domain: SpatialOutputAdapter} decoders.
        num_global_tokens (int): Number of global (CLS-like) tokens. Default 1.
        dim_tokens (int): Encoder token dimensionality. Default 768.
        depth (int): Encoder transformer depth. Default 12.
        num_heads (int): Attention heads. Default 12.
        mlp_ratio (float): MLP hidden-dim ratio. Default 4.0.
        qkv_bias (bool): Use bias in QKV projections. Default True.
        drop_rate (float): Dropout after MLP/Attention. Default 0.0.
        attn_drop_rate (float): Attention-matrix dropout. Default 0.0.
        drop_path_rate (float): DropPath rate. Default 0.0.
        norm_layer (nn.Module): Normalisation layer factory. Default LayerNorm.
        input_domains (list[str]): Ordered list of input domain names.
        output_domains (list[str]): Ordered list of output domain names.
    """

    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 num_global_tokens: int = 1,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 input_domains: List[str] = [],
                 output_domains: List[str] = [],
                 ):
        super().__init__()

        self.input_domains = input_domains
        self.output_domains = output_domains

        # Initialise input and output adapters with encoder token dimension
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)

        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Projections to bridge encoder (dim_tokens) and decoder (decoder_dim) spaces
        decoder_dim = self.output_adapters[self.output_domains[0]].dim_tokens
        self.proj_mask  = nn.Linear(dim_tokens, decoder_dim)
        self.proj_vis   = nn.Linear(dim_tokens, decoder_dim)
        self.layer_norm = nn.LayerNorm(decoder_dim)

        # Global (CLS-like) tokens prepended to the visible token sequence
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        # Transformer encoder with stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder = nn.Sequential(*[
            Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # Treat Q, K, V weight blocks separately (Xavier-uniform per block)
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
            if isinstance(m, nn.Conv2d) and '.proj' in name:
                # From MAE: initialise patch-embedding conv like nn.Linear
                w = m.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}
        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                no_wd_set |= {f'input_adapters.{task}.{n}' for n in adapter.no_weight_decay()}
        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                no_wd_set |= {f'output_adapters.{task}.{n}' for n in adapter.no_weight_decay()}
        return no_wd_set

    def generate_input_info(self, input_task_tokens: Dict[str, torch.Tensor],
                            image_size: tuple) -> OrderedDict:
        """Builds the input_info metadata dict consumed by SpatialOutputAdapter."""
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            input_info['tasks'][domain] = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens
        return input_info

    def generate_eval_masks(self,
                            input_task_tokens: Dict[str, torch.Tensor],
                            num_encoded_tokens: int,
                            real_output_index: List[int]):
        """
        Generates deterministic token visibility masks for evaluation.

        Tokens from domains in `real_output_index` (the stains to impute) are
        all marked as masked (1); tokens from fixed domains and context output
        domains are all marked as visible (0).

        The key design:
        - Fixed domains (e.g. DAPI): all 196 tokens visible.
        - Context output domains (output domains NOT in real_output_index):
          all 196 tokens visible, providing additional spatial context.
        - Target output domains (in real_output_index): all tokens masked,
          forcing the decoder to reconstruct them from context alone.

        `num_encoded_tokens` counts visible tokens from context output domains:
            num_encoded_tokens = 196 * (n_output_domains - len(real_output_index))

        Args:
            input_task_tokens (dict): {domain: (B, N, D)} from input adapters.
            num_encoded_tokens (int): Total visible tokens from context output domains.
            real_output_index (list[int]): Indices into self.output_domains specifying
                which output domains to mask (impute).

        Returns:
            task_masks (dict):   {domain: (B, N)} binary mask. 0 = visible.
            ids_keep   (Tensor): (B, N_visible) global indices for encoder input.
            ids_restore(Tensor): (B, N_total) maps [visible || masked] back to original.
            ids_masked (Tensor): (B, N_masked) global indices for masked tokens.
        """
        B = list(input_task_tokens.values())[0].shape[0]
        device = list(input_task_tokens.values())[0].device

        # Visible token count for each output domain:
        # 196 (all visible) for context domains; 0 for imputation targets
        num_tokens_per_domain = list(input_task_tokens.values())[0].shape[1]  # 196
        samples_per_task = num_tokens_per_domain * torch.ones(
            B, len(self.output_domains), device=device, dtype=torch.long
        )
        samples_per_task[:, real_output_index] = 0

        task_masks = []
        k = 0
        num_tokens_per_task = [t.shape[1] for t in input_task_tokens.values()]

        for i, num_tokens in enumerate(num_tokens_per_task):
            domain = self.input_domains[i]
            if domain not in self.output_domains:
                # Fixed domain: all tokens always visible
                task_masks.append(
                    torch.zeros(B, num_tokens, device=device, dtype=torch.long)
                )
                continue

            # Randomly shuffle token positions so mask boundary is not spatially biased
            noise = torch.rand(B, num_tokens, device=device)
            ids_arange_shuffle = torch.argsort(noise, dim=1)
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # Tokens ranked below samples_per_task are visible (0); the rest are masked (1)
            mask = torch.where(mask < samples_per_task[:, k].unsqueeze(1), 0, 1)
            k += 1
            task_masks.append(mask)

        n_fixed = len(self.input_domains) - len(self.output_domains)
        n_visible = num_encoded_tokens + num_tokens_per_domain * n_fixed

        mask_all = torch.cat(task_masks, dim=1)
        # Add small noise to break ties between equally-ranked visible tokens
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep   = ids_shuffle[:, :n_visible]
        ids_masked = ids_shuffle[:, n_visible:]

        # Recompute binary mask to ensure exact consistency with ids_keep / ids_masked
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :n_visible] = 0
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        task_masks_split = torch.split(mask_all, num_tokens_per_task, dim=1)
        task_masks = {
            domain: mask
            for domain, mask in zip(input_task_tokens.keys(), task_masks_split)
        }

        return task_masks, ids_keep, ids_restore, ids_masked

    def forward(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                num_encoded_tokens: int = 0,
                real_output_index: Optional[List[int]] = None,
                fp32_output_adapters: List[str] = []):
        """
        Evaluation forward pass with selective domain masking.

        Args:
            x (dict): {domain: (B, 1, H, W)}.  Domains listed in
                `real_output_index` should be zeroed by the caller before
                this call to simulate absent stains.
            num_encoded_tokens (int): Total visible tokens from context output
                domains. Should be 196 * (n_output - len(real_output_index)).
                Pass 0 when all output domains are being imputed.
            real_output_index (list[int] | None): Indices into self.output_domains
                for the stains to impute. Defaults to all output domains.
            fp32_output_adapters (list[str]): Domains whose output adapter should
                run in full float32 (excluded from preds in this call).

        Returns:
            preds      (dict): {domain: (B, 1, H, W)} reconstructed stain images
                               for each domain in real_output_index.
            task_masks (dict): {domain: (B, N)} binary mask (0 = visible to encoder).
        """
        if real_output_index is None:
            real_output_index = list(range(len(self.output_domains)))

        B, C, H, W = list(x.values())[0].shape

        # Tokenise each input domain through its adapter
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }
        input_info = self.generate_input_info(
            input_task_tokens=input_task_tokens, image_size=(H, W)
        )

        # Build deterministic visibility masks
        task_masks, ids_keep, ids_restore, ids_masked = self.generate_eval_masks(
            input_task_tokens, num_encoded_tokens, real_output_index
        )

        # Concatenate all domain tokens then select visible / masked subsets
        input_tokens = torch.cat(list(input_task_tokens.values()), dim=1)

        visual_tokens = torch.gather(
            input_tokens, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, input_tokens.shape[2])
        )
        mask_tokens = torch.gather(
            input_tokens, dim=1,
            index=ids_masked.unsqueeze(-1).expand(-1, -1, input_tokens.shape[2])
        )

        # Prepend global (CLS-like) tokens to the visible sequence
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        visual_tokens = torch.cat([visual_tokens, global_tokens], dim=1)

        # Project masked tokens into the decoder dimension
        mask_tokens = self.proj_mask(mask_tokens)
        mask_tokens = self.layer_norm(mask_tokens)

        # Encode the visible tokens
        encoder_tokens = self.encoder(visual_tokens)
        encoder_tokens = self.proj_vis(encoder_tokens)
        encoder_tokens = self.layer_norm(encoder_tokens)

        if self.output_adapters is None:
            return encoder_tokens, task_masks

        # Decode only the imputation-target domains
        decode_domains = [self.output_domains[i] for i in real_output_index]
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
                mask_tokens=mask_tokens,
            )
            for domain in decode_domains
            if domain not in fp32_output_adapters
        }

        return preds, task_masks


def pretrain_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs) -> MultiMAE:
    """
    Builds a base-scale MultiMAE evaluation model (ViT-B/16 configuration).

    Uses 768-dim tokens, 12 transformer layers, and 12 attention heads.

    Args:
        input_adapters  (dict): {domain: PatchedInputAdapter instance}.
        output_adapters (dict): {domain: SpatialOutputAdapter instance}.
        **kwargs: Passed to MultiMAE (e.g. input_domains, output_domains).

    Returns:
        MultiMAE: Initialised evaluation model.
    """
    return MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def pretrain_multimae_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs) -> MultiMAE:
    """
    Builds a large-scale MultiMAE evaluation model (ViT-L/16 configuration).

    Uses 1024-dim tokens, 24 transformer layers, and 16 attention heads.

    Args:
        input_adapters  (dict): {domain: PatchedInputAdapter instance}.
        output_adapters (dict): {domain: SpatialOutputAdapter instance}.
        **kwargs: Passed to MultiMAE (e.g. input_domains, output_domains).

    Returns:
        MultiMAE: Initialised large evaluation model.
    """
    return MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
