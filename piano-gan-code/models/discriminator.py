"""
models/discriminator.py — Axis-Decomposed Piano GAN Discriminator (WGAN-GP Critic)

Architecture (full model)
─────────────────────────
  x: (B, 1, 128, 88)

    → EntryConvBlock(c_out=12)      (B,  1, 128, 88) → (B, 12, 128, 88)
    → DownsampleBlock(12 → 16)    
    → DownsampleBlock(16 → 24)    
    → DownsampleBlock(24 → 48)    

  → GlobalAvgPool  (B, 48, 8, 88) → (B, 48, 1, 1) → flatten → (B, 48)
  → Linear(48 → 128) + LeakyReLU(0.2)
  → Linear(128 → 1)                           ← raw WGAN-GP critic score, NO sigmoid

Normalization: LayerNorm (GroupNorm(1, C)) everywhere — NEVER BatchNorm.
  BatchNorm accumulates batch statistics; the WGAN-GP gradient penalty is
  computed independently for each sample in an interpolated set, so batch-
  level statistics would contaminate the per-sample gradient norm. LayerNorm
  is strictly per-sample and does not violate the 1-Lipschitz constraint.

On Global Average Pooling:
  GAP is applied to the output of 4 stages of axis-decomposed convolution,
  NOT to the raw piano roll. By this point each of the 48 channels encodes
  a learned, non-linear, spatially-aware feature (e.g. density of third-
  interval activations over beat positions). GAP on deep feature maps is
  identical to what ResNets, BigGAN, and StyleGAN discriminators do before
  their final classification layer. It is applying GAP to RAW input activations
  that destroys spatial structure — that concern does not apply here.
  The WGAN-GP gradient penalty flows correctly through GAP as it is
  differentiable w.r.t. the interpolated input x_hat.

Ablation flags: same as Generator (use_temporal, use_pitch control the
  discriminator’s DownsampleBlocks symmetrically).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from models.blocks import EntryConvBlock, DownsampleBlock


# ── Weight Initialization ──────────────────────────────────────────────────
def init_weights(m: nn.Module):
    """
    Applies standard GAN weight initialization to prevent massive pre-activation
    variance from triggering catastrophic gradient penalty spikes at Step 0.
    """
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        # If wrapped in spectral_norm, the parameter is renamed to 'weight_orig'
        weight_param = getattr(m, 'weight_orig', getattr(m, 'weight', None))
        if weight_param is not None:
            nn.init.normal_(weight_param.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Discriminator(nn.Module):
    """
    Axis-Decomposed Piano GAN Discriminator (WGAN-GP Critic).

    Parameters
    ----------
    use_temporal : bool  Include temporal (time-axis 1D conv) branch.
    use_pitch    : bool  Include pitch-axis (dilated 1D conv) branch.
    """

    # Channel schedule (mirrors generator in reverse): 12 → 16 → 24 → 32 → 48
    # Start at c_out=12 so the first DownsampleBlock gets c_in=12:
    #   PitchBlock base = 12 // 5 = 2  →  branches = [2,2,2,2,4]  (defensible)
    # Raised from [8,12,16,24,32] where c_in=8 gave base=1 (indefensible for the paper).
    CHANNEL_SCHEDULE = [12, 16, 24, 48]

    def __init__(self,
                 use_temporal: bool = True,
                 use_pitch:    bool = True):
        super().__init__()

        # ── Entry: multi-scale 2D conv, 1 → 12 channels ───────────────────────
        self.entry  = EntryConvBlock(c_out=self.CHANNEL_SCHEDULE[0])

        # ── Three downsampling stages ────────────
        self.blocks = nn.ModuleList()
        schedule    = self.CHANNEL_SCHEDULE
        for i in range(len(schedule) - 1):
            self.blocks.append(
                DownsampleBlock(
                    c_in         = schedule[i],
                    c_out        = schedule[i + 1],
                    use_temporal = use_temporal,
                    use_pitch    = use_pitch,
                )
            )

        # ── Global Average Pooling + small MLP head ───────────────────────────
        # After 4 downsamples: (B, 48, 8, 88).
        # GAP collapses each channel's spatial map → scalar (B, 48).
        # Applied to deep learned features, NOT raw input — see module docstring.
        self.gap    = nn.AdaptiveAvgPool2d(1)              # (B, C, 1, 1)
        c_last      = schedule[-1]                          # 48
        self.fc1    = spectral_norm(nn.Linear(c_last, 128))
        self.fc2    = spectral_norm(nn.Linear(128, 1))
        # NO final activation: raw critic score for WGAN-GP

        # ── Initialize weights safely BEFORE training ─────────────────────────
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 128, 88)  real or generated piano roll in [-1, 1]

        Returns
        -------
        score : (B, 1)  raw WGAN-GP critic value (higher = more real)
        """
        x = self.entry(x)                        # (B, 12, 128, 88)

        for block in self.blocks:
            x = block(x)                         # (B, 48, 8, 88) at end

        x = self.gap(x).flatten(start_dim=1)     # (B, 48)
        x = F.leaky_relu(self.fc1(x), 0.2)       # (B, 128)
        x = self.fc2(x)                          # (B, 1)
        return x

    # ── Convenience ──────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        s = super().__repr__()
        return f"{s}\n[Discriminator]  Trainable params: {self.num_parameters():,}"
