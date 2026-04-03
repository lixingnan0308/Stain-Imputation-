"""
unet.py — OS2CRDiff model architecture.

Paper: "OS2CR-Diff: A Self-Refining Diffusion Framework for CD8 Imputation
        from One-Step Inference to Conditional Representation"
Li et al., IEEE BIBM 2025.  https://ieeexplore.ieee.org/document/11356570/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class ResNetToSize(nn.Module):
    """
    ResNet-style downsampler that maps a 224×224 input to a specified target size.

    Applies an initial 7×7 strided conv (224→112) followed by additional
    stride-2 residual blocks until the target resolution is reached,
    then adjusts the channel count with a 1×1 projection.
    """

    def __init__(self, in_channels=1, target_size=56, out_channels=256):
        super().__init__()
        self.target_size  = target_size
        self.out_channels = out_channels

        downsample_factor = 224 // target_size
        num_downsamples   = int(math.log2(downsample_factor)) if downsample_factor > 1 else 0

        layers = []
        current_channels = in_channels

        # Initial conv: 224 → 112.
        layers.extend([
            nn.Conv2d(current_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ])
        current_channels = 64

        # Additional residual downsampling blocks (-1 because the initial conv already halved).
        for _ in range(num_downsamples - 1):
            next_channels = min(current_channels * 2, 512)
            layers.extend([
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(next_channels, next_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(next_channels, next_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = next_channels

        # Channel projection to the desired output width.
        if current_channels != out_channels:
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (B, in_channels, 224, 224).

        Returns:
            torch.Tensor: Shape (B, out_channels, target_size, target_size).
        """
        out = self.network(x)
        if out.size(-1) != self.target_size:
            out = F.interpolate(out, size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Query projection (from input1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        
        # Key and Value projections (from input2) 
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Query source, shape (B, N, dim).
            x2 (torch.Tensor): Key/value source, shape (B, N, dim).

        Returns:
            torch.Tensor: Shape (B, N, dim).
        """
        b, n, d = x1.shape

        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        return self.to_out(out)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.0):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x1, x2):
        attn_out = self.cross_attn(self.norm1(x1), self.norm1(x2))
        x1 = x1 + attn_out
        x1 = x1 + self.ffn(self.norm2(x1))
        return self.norm1(x1)

class SharedMultiScaleConditionEncoder(nn.Module):
    """
    Shared multi-scale condition encoder.

    Encodes three condition streams (prior CD8, fixed inputs, and conditional
    biomarkers) at each required UNet resolution, fuses them via cross-attention
    and adaptive spatial-channel gating, and produces per-block feature maps
    that are injected into the UNet at the specified condition blocks.
    """

    def __init__(self, block_specs):
        """
        Args:
            block_specs (list): [(block_name, spatial_size, out_channels), ...]
        """
        super().__init__()
        self.block_specs = block_specs

        block_names  = list(set(name for name, _, _ in block_specs))
        base_channels = 512  # unified output channels for all shared encoders

        self.shared_condition1_encoders = nn.ModuleDict()
        self.shared_condition2_encoders = nn.ModuleDict()
        self.shared_condition3_encoders = nn.ModuleDict()
        self.cross_attention_1_2        = nn.ModuleDict()
        self.cross_attention_1_3        = nn.ModuleDict()
        
        for block_name in block_names:
            key      = str(block_name)
            size     = [s for n, s, _ in self.block_specs if n == block_name][0]
            size_key = f'{key}-{size}'

            self.shared_condition1_encoders[size_key] = ResNetToSize(
                in_channels=1, target_size=size, out_channels=base_channels)
            self.shared_condition2_encoders[size_key] = ResNetToSize(
                in_channels=2, target_size=size, out_channels=base_channels)
            self.shared_condition3_encoders[size_key] = ResNetToSize(
                in_channels=3, target_size=size, out_channels=base_channels)

            self.cross_attention_1_2[size_key] = CrossAttention(dim=base_channels, num_heads=4)
            self.cross_attention_1_3[size_key] = CrossAttention(dim=base_channels, num_heads=4)

        # Channel adapters and fusion layers per block.
        self.channel_adapters = nn.ModuleDict()
        self.fusion_layers     = nn.ModuleDict()

        for block_name, size, target_channels in block_specs:
            size_key = f'{block_name}-{size}'
            self.channel_adapters[size_key] = nn.Sequential(
                nn.Conv2d(base_channels, target_channels, kernel_size=1),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True),
            )
            self.fusion_layers[size_key] = nn.Sequential(
                nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(target_channels, target_channels, kernel_size=1),
            )

        # Adaptive spatial-channel gating (SE-style) over the 5-stream concat.
        self.gating_layers = nn.ModuleDict()
        for block_name, size, _ in block_specs:
            size_key = f'{block_name}-{size}'
            self.gating_layers[size_key] = nn.Sequential(
                nn.Conv2d(base_channels * 5, base_channels // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 2, 5, kernel_size=1),
            )

            
    
    def forward(self, condition1, condition2, condition3):
        """
        Encodes the three condition streams and returns per-block feature maps.

        Args:
            condition1 (torch.Tensor): Prior CD8 map, shape (B, 1, 224, 224).
            condition2 (torch.Tensor): Fixed inputs (DAPI + AF), shape (B, 2, 224, 224).
            condition3 (torch.Tensor): Conditional biomarkers, shape (B, 3, 224, 224).

        Returns:
            dict[str, torch.Tensor]: Mapping from '{block_name}-{size}' to the
                                     fused condition feature of shape (B, C, size, size).
        """
        block_names     = list(set(name for name, _, _ in self.block_specs))
        shared_features = {}

        for block_name in block_names:
            key      = str(block_name)
            size     = [s for n, s, _ in self.block_specs if n == block_name][0]
            size_key = f'{key}-{size}'

            feat1 = self.shared_condition1_encoders[size_key](condition1)
            feat2 = self.shared_condition2_encoders[size_key](condition2)
            feat3 = self.shared_condition3_encoders[size_key](condition3)

            def to_seq(f):
                return f.view(f.size(0), f.size(1), -1).permute(0, 2, 1)

            def from_seq(f, ref):
                return f.permute(0, 2, 1).view(ref.size(0), ref.size(1),
                                                ref.size(2), ref.size(3))

            cross_1_2 = from_seq(self.cross_attention_1_2[size_key](to_seq(feat1), to_seq(feat2)), feat1)
            cross_1_3 = from_seq(self.cross_attention_1_3[size_key](to_seq(feat1), to_seq(feat3)), feat1)

            # Adaptive gating over 5 streams: feat1, feat2, feat3, cross_1_2, cross_1_3.
            concat  = torch.cat([feat1, feat2, feat3, cross_1_2, cross_1_3], dim=1)
            weights = F.softmax(self.gating_layers[size_key](concat), dim=1)
            fused   = (weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2
                       + weights[:, 2:3] * feat3 + weights[:, 3:4] * cross_1_2
                       + weights[:, 4:5] * cross_1_3)
            shared_features[size_key] = F.layer_norm(fused, fused.shape[-1:])

        condition_features = {}
        for block_name in block_names:
            key      = str(block_name)
            size     = [s for n, s, _ in self.block_specs if n == block_name][0]
            size_key = f'{key}-{size}'
            condition_features[size_key] = self.fusion_layers[size_key](
                self.channel_adapters[size_key](shared_features[size_key])
            )

        return condition_features


class DownBlock(nn.Module):
    r"""
    Down conv block with optional attention and condition fusion.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample=True, num_heads=4, num_layers=1, use_attention=True, 
                 use_condition=False):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.use_attention = use_attention
        self.use_condition = use_condition
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.use_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, out_channels)
                 for _ in range(num_layers)]
            )
            
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb, condition_features=None, block_name=None):
        out = x
        for i in range(self.num_layers):
            
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self-attention block of Unet (optional)
            if self.use_attention:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        
        # Inject condition features before downsampling.
        if self.use_condition and condition_features is not None and block_name is not None:
            if block_name in condition_features:
                out = out + condition_features[block_name]

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention and condition fusion.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1, 
                 use_condition=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_condition = use_condition
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers+1)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
    
    def forward(self, x, t_emb, condition_features=None, block_name=None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        # Inject condition features after the mid-block residual path.
        if self.use_condition and condition_features is not None and block_name is not None:
            if block_name in condition_features:
                out = out + condition_features[block_name]

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with optional attention and condition fusion.
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1, 
                 use_attention=True, use_condition=False):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.use_attention = use_attention
        self.use_condition = use_condition
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.use_attention:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(8, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        # up_sample_conv is set externally by OS2CRDiff.__init__.
        self.up_sample_conv = None

    def forward(self, x, out_down, t_emb, condition_features=None, block_name=None):
        if self.up_sample_conv is not None:
            x = self.up_sample_conv(x)
        
        # concat skip connection
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.use_attention:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        
        # Inject condition features after the up-block residual path.
        if self.use_condition and condition_features is not None and block_name is not None:
            if block_name in condition_features:
                out = out + condition_features[block_name]

        return out


class OS2CRDiff(nn.Module):
    """
    OS2CR-Diff conditional UNet with shared multi-scale condition encoder.

    Implements the denoising network of OS2CR-Diff.  Three condition streams
    (prior CD8, fixed inputs, conditional biomarkers) are encoded at multiple
    spatial resolutions and injected into the UNet via additive feature fusion
    at the blocks listed in ``model_config['condition_blocks']``.
    """
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.num_heads = model_config.get('num_heads', 4)
        self.use_condition = model_config.get('use_condition', True)
        
        self.use_attention    = model_config.get('use_attention', [True] * (len(self.down_channels) - 1))
        self.condition_blocks = model_config.get('condition_blocks', [2, 'mid1', 'mid2', 0, 1])

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample)   == len(self.down_channels) - 1
        assert len(self.use_attention) == len(self.down_channels) - 1

        if self.use_condition:
            self.block_specs = self._calculate_block_specs()
            self.condition_encoder = SharedMultiScaleConditionEncoder(self.block_specs)
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.up_attention = list(reversed(self.use_attention))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Down blocks.
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(
                self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                down_sample=self.down_sample[i], num_layers=self.num_down_layers,
                num_heads=self.num_heads, use_attention=self.use_attention[i],
                use_condition=f'down_{i}' in self.condition_blocks,
            ))

        # Mid blocks.
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(
                self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                num_layers=self.num_mid_layers, num_heads=self.num_heads,
                use_condition=f'mid{i + 1}' in self.condition_blocks,
            ))

        # Up blocks.
        # Channel sizes (with default config [64, 128, 256, 256, 512], mid [512, 512, 256]):
        #   ups[0]: (mid_out=256 + skip=256) → 256
        #   ups[1]: (256 + skip=256)          → 128
        #   ups[2]: (128 + skip=128)          → 64
        #   ups[3]: (64  + skip=64)           → 64
        up_configs = [
            (self.mid_channels[-1], self.down_channels[-2], 256),
            (256,                   self.down_channels[-3], 128),
            (128,                   self.down_channels[-4], 64),
            (64,                    self.down_channels[0],  64),
        ]

        self.ups = nn.ModuleList([])
        for i, (prev_ch, skip_ch, out_ch) in enumerate(up_configs):
            if i >= len(self.down_channels) - 1:
                break
            up_block = UpBlock(
                prev_ch + skip_ch, out_ch, self.t_emb_dim,
                up_sample=self.down_sample[-(i + 1)], num_layers=self.num_up_layers,
                num_heads=self.num_heads, use_attention=self.use_attention[-(i + 1)],
                use_condition=f'up_{i}' in self.condition_blocks,
            )
            up_block.up_sample_conv = (
                nn.ConvTranspose2d(prev_ch, prev_ch, 4, 2, 1)
                if self.down_sample[-(i + 1)] else nn.Identity()
            )
            self.ups.append(up_block)

        final_ch = up_configs[len(self.ups) - 1][2] if self.ups else 64
        
        self.norm_out = nn.GroupNorm(8, final_ch)
        self.conv_out = nn.Conv2d(final_ch, im_channels, kernel_size=3, padding=1)
    
    def _calculate_block_specs(self):
        """
        Derives the (block_name, spatial_size, channels) spec for each condition block.

        Spatial sizes follow the UNet resolution changes:
            down path: 224 → 112 → 56 → 28 → 14
            up path:    14 → 28 → 56 → 112 → 224

        Returns:
            list[tuple]: [(block_name, size, channels), ...]
        """
        block_specs = []

        current_size = 224
        sizes = [current_size]
        for ds in self.down_sample:
            if ds:
                current_size //= 2
            sizes.append(current_size)

        up_sizes    = [14, 28, 56, 112, 224]
        up_channels = [256, 128, 64, 64]

        for block_spec in self.condition_blocks:
            if block_spec == 'mid1':
                block_specs.append(('mid_1', 14, self.mid_channels[1]))

            elif block_spec == 'mid2':
                ch = self.mid_channels[2] if len(self.mid_channels) > 2 else self.mid_channels[1]
                block_specs.append(('mid_2', 14, ch))

            elif isinstance(block_spec, str):
                if block_spec.startswith('down_'):
                    idx  = int(block_spec.split('_')[1])
                    block_specs.append((f'down_{idx}', sizes[idx], self.down_channels[idx + 1]))

                elif block_spec.startswith('up_'):
                    idx  = int(block_spec.split('_')[1])
                    block_specs.append((f'up_{idx}', up_sizes[idx + 1], up_channels[idx]))

        return block_specs
    
    def forward(self, x, t, condition1=None, condition2=None, condition3=None):
        """
        Forward pass of OS2CR-Diff's conditional UNet.

        Args:
            x          (torch.Tensor): Noisy image, shape (B, C, H, W).
            t          (int | torch.Tensor): Timestep(s).
            condition1 (torch.Tensor | None): Prior CD8, shape (B, 1, 224, 224).
            condition2 (torch.Tensor | None): Fixed inputs, shape (B, 2, 224, 224).
            condition3 (torch.Tensor | None): Conditional biomarkers, shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: v-prediction of shape (B, C, H, W).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x.shape[0])

        condition_features = None
        if self.use_condition and condition1 is not None and condition2 is not None:
            condition_features = self.condition_encoder(condition1, condition2, condition3)

        out   = self.conv_in(x)
        t_emb = self.t_proj(get_time_embedding(t.long(), self.t_emb_dim))

        # Encoder (down) path.
        down_outs = []
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            bname = f'down_{idx}' if f'down_{idx}' in self.condition_blocks else None
            key   = f'{bname}-{[s for n, s, _ in self.block_specs if n == bname][0]}' \
                    if bname else None
            out   = down(out, t_emb, condition_features, key)

        # Bottleneck (mid) path.
        for idx, mid in enumerate(self.mids):
            bname = f'mid_{idx + 1}' if f'mid{idx + 1}' in self.condition_blocks else None
            key   = f'{bname}-{[s for n, s, _ in self.block_specs if n == bname][0]}' \
                    if bname else None
            out   = mid(out, t_emb, condition_features, key)

        # Decoder (up) path.
        for idx, up in enumerate(self.ups):
            down_out = down_outs.pop()
            bname    = f'up_{idx}' if f'up_{idx}' in self.condition_blocks else None
            key      = f'{bname}_{[s for n, s, _ in self.block_specs if n == bname][0]}' \
                       if bname else None
            out      = up(out, down_out, t_emb, condition_features, key)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out


