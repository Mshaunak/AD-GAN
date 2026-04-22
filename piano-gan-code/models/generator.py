"""
models/generator.py — Axis-Decomposed Piano GAN Generator

Architecture (full model)
─────────────────────────
  z ~ N(0, 1)  shape (B, 64)
  → Linear(64 → 48·8·88 = 33,792) → reshape (B, 48, 8, 88)

  → UpsampleBlock(48 → 32)   T:  8 →  16
  → UpsampleBlock(32 → 24)   T: 16 →  32
  → UpsampleBlock(24 → 16)   T: 32 →  64
  → UpsampleBlock(16 → 12)   T: 64 → 128

  → EnhancedFusionBlock(c_in=12)   (B, 12, 128, 88) → (B, 1, 128, 88)

Output values ∈ [-1, 1] via Tanh.  Training data is normalised to [-1, 1]
so that the generator and discriminator live in the same dynamic range.

Ablation flags
──────────────
  use_temporal   — include temporal (time-axis) branch in every UpsampleBlock
  use_pitch      — include pitch-axis branch  in every UpsampleBlock
  use_fusion     — use EnhancedFusionBlock head; if False, use a plain 1×1 conv + Tanh
  (When both use_temporal=False and use_pitch=False the blocks degrade to
   a MuseGAN-style plain 2D conv baseline.)
"""

import torch
import torch.nn as nn
from models.blocks import UpsampleBlock, EnhancedFusionBlock


# ── Weight Initialization ──────────────────────────────────────────────────
def init_weights(m: nn.Module):
    """
    Applies standard GAN weight initialization to prevent massive pre-activation
    variance from saturating Tanh/ReLU layers and causing mode collapse.
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


class Generator(nn.Module):
    """
    Axis-Decomposed Piano GAN Generator.

    Parameters
    ----------
    z_dim        : int   Dimensionality of the input noise vector.
    use_temporal : bool  Include temporal (time-axis 1D conv) branch.
    use_pitch    : bool  Include pitch-axis (dilated 1D conv) branch.
    use_fusion   : bool  Use EnhancedFusionBlock output head (True) or a
                         plain Conv1×1 + Tanh baseline (False for ablation).
    """

    # Channel schedule: 48 → 32 → 24 → 16 → 12
    # Minimum c_in to any PitchBlock = 12 (base = 12//5 = 2: [2,2,2,2,4]).
    # Raised from [32,24,16,12,8] to prevent musically-motivated pitch branches
    # from collapsing to 1 channel, which would make the harmonic-interval claim
    # indefensible. FC cost: Linear(64 → 33,792) ≈ 2.2M (vs 23M in v1).
    CHANNEL_SCHEDULE = [48, 32, 24, 12]

    def __init__(self,
                 z_dim:        int  = 64,
                 use_temporal: bool = True,
                 use_pitch:    bool = True,
                 use_fusion:   bool = True):
        super().__init__()

        self.z_dim = z_dim
        c0         = self.CHANNEL_SCHEDULE[0]   # 48

        # ── Latent → spatial ────────────────────────────────────────────────
        # Maps z ∈ R^64 to (B, 48, 8, 88): 8 coarse time frames × 88 pitches
        from torch.nn.utils.parametrizations import spectral_norm
        self.fc = spectral_norm(nn.Linear(z_dim, c0 * 8 * 88))

        # ── Three upsampling stages (T: 8 → 16 → 32 → 64) ────────────
        self.blocks = nn.ModuleList()
        schedule    = self.CHANNEL_SCHEDULE
        for i in range(len(schedule) - 1):
            self.blocks.append(
                UpsampleBlock(
                    c_in         = schedule[i],
                    c_out        = schedule[i + 1],
                    use_temporal = use_temporal,
                    use_pitch    = use_pitch,
                )
            )

        # ── Output head ──────────────────────────────────────────────────────
        c_final = schedule[-1]   # 12
        if use_fusion:
            self.head = EnhancedFusionBlock(c_in=c_final)
        else:
            self.head = nn.Sequential(
                nn.Conv2d(c_final, 1, kernel_size=1, bias=True),
                nn.Tanh(),
            )
            
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, z_dim)

        Returns
        -------
        x : (B, 1, 128, 88)  values in [-1, 1]
        """
        B  = z.size(0)
        x  = self.fc(z)
        x  = x.view(B, self.CHANNEL_SCHEDULE[0], 8, 88)  # (B, 48, 8, 88)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)     # (B, 1, 128, 88)
        return x

    # ── Convenience ──────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        s = super().__repr__()
        return f"{s}\n[Generator]  Trainable params: {self.num_parameters():,}"
