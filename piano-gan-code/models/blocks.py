import torch
import torch.nn as nn
import torch.nn.functional as F

# ═════════════════════════════════════════════════════════════════════════════
# Normalization
# ═════════════════════════════════════════════════════════════════════════════

def _norm1d(norm_type: str, num_features: int) -> nn.Module:
    """Return a 1-D normalisation layer matching norm_type."""
    if norm_type == "bn":
        return nn.BatchNorm1d(num_features)
    elif norm_type == "ln":
        return nn.GroupNorm(1, num_features)
    return nn.Identity()


def _norm2d(norm_type: str, num_features: int) -> nn.Module:
    """Return a 2-D normalisation layer matching norm_type."""
    if norm_type == "bn":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "ln":
        return nn.GroupNorm(1, num_features)
    return nn.Identity()


def _act(act_type: str) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "leaky":
        return nn.LeakyReLU(0.2, inplace=True)
    return nn.Identity()


# ══════════════════
# TemporalBlock
# ══════════════════

class TemporalBlock(nn.Module):
    """
    1D convolution along the TIME axis only.
    Each pitch position is processed independently (P is folded into batch).

    Input  shape: (B, C_in, T, P)
    Output shape: (B, C_out, T', P)
      where T' = T when stride=1  (generator)
            T' = ceil(T/2) when stride=2  (discriminator)

    Kernel size 5 with padding 2 → no time-step lost at stride=1.
    At stride=2 this halves the time dimension for the discriminator.
    """

    def __init__(self,c_in: int, c_out: int, norm:  str = "bn", activation: str = "relu", stride: int = 1):
        super().__init__()
        self.c_out = c_out
        padding    = 2 if stride == 1 else 2

        self.conv  = nn.Conv2d(c_in, c_out,
                               kernel_size=(5, 1),
                               stride=(stride, 1),
                               padding=(padding, 0),
                               bias=False)
        self.norm  = _norm2d(norm, c_out)
        self.act   = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x) 
        x = self.norm(x)
        x = self.act(x)
        return x


# ═══════════════
# PitchBlock 
# ═══════════════

class PitchBlock(nn.Module):

    DILATIONS = [1, 3, 4, 7, 12]

    def __init__(self,
                 c_in:       int,
                 c_out:      int,
                 norm:       str = "bn",
                 activation: str = "relu"):
        super().__init__()
        self.c_out = c_out

        base         = c_in // 5
        branch_sizes = [base] * 4 + [c_in - 4 * base]

        self.branches = nn.ModuleList()
        for d, bs in zip(self.DILATIONS, branch_sizes):
            
            self.branches.append(
                nn.Conv2d(c_in, bs,
                          kernel_size=(1, 3),
                          dilation=(1, d),
                          padding=(0, d),
                          bias=False)
            )

        self.mix  = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.norm = _norm2d(norm, c_out)
        self.act  = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, P)
        branch_outs = [br(x) for br in self.branches]
        x           = torch.cat(branch_outs, dim=1)        # (B, C_in, T, P)

        x = self.mix(x)                                    # (B, C_out, T, P)
        x = self.norm(x)
        x = self.act(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# MergeBlock — learned combination of temporal + pitch branches
# ═════════════════════════════════════════════════════════════════════════════

class MergeBlock(nn.Module):
    """
    Merges temporal branch output Yt and pitch branch output Yp via:

        Y = Norm(ReLU(Conv1×1([Yt ; Yp])))

    Concatenation followed by 1×1 projection is strictly more expressive than
    addition (Y = Yt + Yp, which has zero-sum coupling) or weighted gating
    (which still imposes a sum-to-one constraint).

    Both inputs must have the same spatial size (T, P) and the same number
    of channels C.  Output also has C channels.

    Input  shapes: Yt (B, C, T, P),  Yp (B, C, T, P)
    Output shape:  (B, C, T, P)
    """

    def __init__(self, c: int, norm: str, activation: str):
        super().__init__()
        self.conv  = nn.Conv2d(2 * c, c, kernel_size=1, bias=False)
        self.norm  = _norm2d(norm, c)
        self.act   = _act(activation)

    def forward(self, yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        x = torch.cat([yt, yp], dim=1)    # (B, 2C, T, P)
        x = self.conv(x)                  # (B,  C, T, P)
        x = self.norm(x)
        x = self.act(x)
        return x


# ═════════════════
# UpsampleBloc
# ═════════════════

class UpsampleBlock(nn.Module):
    """
    Generator block.  Doubles the time dimension via nearest-neighbour
    upsampling followed by 1D convolutions on each decomposed axis.

    When both branches are disabled (baseline_2d ablation) falls back to a
    plain 2D Conv (kernel 3×3) on the upsampled tensor.

    Input  shape: (B, C_in, T,  P)
    Output shape: (B, C_out, 2T, P)
    """

    def __init__(self,
                 c_in:         int,
                 c_out:        int,
                 use_temporal: bool = True,
                 use_pitch:    bool = True):
        super().__init__()
        self.use_temporal   = use_temporal
        self.use_pitch      = use_pitch
        self._fallback_2d   = not use_temporal and not use_pitch

        if self._fallback_2d:
            self.conv2d = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        else:
            if use_temporal:
                self.temporal = TemporalBlock(
                    c_in, c_out, norm="bn", activation="relu", stride=1
                )
            if use_pitch:
                self.pitch = PitchBlock(
                    c_in, c_out, norm="bn", activation="relu"
                )

            if use_temporal and use_pitch:
                self.merge = MergeBlock(c_out, norm="bn", activation="relu")
            else:
                self.proj = nn.Sequential(
                    nn.Conv2d(c_out, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample time first (applies to all paths)
        x_up = F.interpolate(x, scale_factor=(2, 1), mode="nearest")  # (B,C_in,2T,P)

        if self._fallback_2d:
            return self.conv2d(x_up)

        if self.use_temporal and self.use_pitch:
            yt = self.temporal(x_up)                                    # (B,C_out,2T,P)
            yp = self.pitch(x)                                          # (B,C_out,T, P)
            yp = F.interpolate(yp, scale_factor=(2, 1), mode="nearest") # (B,C_out,2T,P)
            return self.merge(yt, yp)

        elif self.use_temporal:
            yt = self.temporal(x_up)
            return self.proj(yt)

        else:   # use_pitch only
            yp = self.pitch(x)
            yp = F.interpolate(yp, scale_factor=(2, 1), mode="nearest")
            return self.proj(yp)


# ═════════════════════
# DownsampleBlock
# ═════════════════════

class DownsampleBlock(nn.Module):
    """
    Discriminator block.  Halves the time dimension using stride-2 temporal
    conv and average pooling on the pitch branch.

    Uses LayerNorm (GroupNorm(1, C)) throughout — see module docstring for
    why BatchNorm is incompatible with WGAN-GP.

    Input  shape: (B, C_in, T,   P)
    Output shape: (B, C_out, T//2, P)
    """

    def __init__(self,
                 c_in:         int,
                 c_out:        int,
                 use_temporal: bool = True,
                 use_pitch:    bool = True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_pitch    = use_pitch
        self._fallback_2d = not use_temporal and not use_pitch

        if self._fallback_2d:
            # ── Baseline 2D conv with stride ─────────────────────────────────
            self.conv2d = nn.Sequential(
                nn.Conv2d(c_in, c_out,
                          kernel_size=3, stride=(2, 1), padding=1, bias=False),
                nn.GroupNorm(1, c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            if use_temporal:
                self.temporal = TemporalBlock(
                    c_in, c_out, norm="ln", activation="leaky", stride=2
                )
            if use_pitch:
                self.pitch = PitchBlock(
                    c_in, c_out, norm="ln", activation="leaky"
                )

            if use_temporal and use_pitch:
                self.merge = MergeBlock(c_out, norm="ln", activation="leaky")
            else:
                self.proj = nn.Sequential(
                    nn.Conv2d(c_out, c_out, kernel_size=1, bias=False),
                    nn.GroupNorm(1, c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fallback_2d:
            return self.conv2d(x)

        if self.use_temporal and self.use_pitch:
            yt = self.temporal(x)                                        
            yp = self.pitch(x)                                           
            yp = F.avg_pool2d(yp, kernel_size=(2, 1), stride=(2, 1))    
            return self.merge(yt, yp)

        elif self.use_temporal:
            yt = self.temporal(x)
            return self.proj(yt)

        else:   # use_pitch only
            yp = self.pitch(x)
            yp = F.avg_pool2d(yp, kernel_size=(2, 1), stride=(2, 1))
            return self.proj(yp)


# ═════════════════════════════════════════════════════════════════════════════
# SqueezeExcitation — channel-wise attention
# ═════════════════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    """
    Global channel reweighting via SE attention.
    Global average pool → 2-layer MLP → sigmoid → per-channel scale.

    Reduction ratio defaults to 4 (reduces to C//4 in the bottleneck).
    Works on any (B, C, H, W) tensor.
    """

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        r = max(c // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = self.pool(x)             # (B, C, 1, 1)
        w = self.fc(w)               # (B, C)
        w = w.view(x.size(0), x.size(1), 1, 1)
        return x * w


# ═════════════════════════════════════════════════════════════════════════════
# EnhancedFusionBlock — Stage 3 generator output head
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedFusionBlock(nn.Module):
    """
    The architectural centrepiece.  Fuses the 1D-decomposed representations
    into a full 2D piano roll using three parallel asymmetric 2D conv branches:

      Branch A  (3×3) — local voice leading: captures adjacent semitone
                         movement and neighbouring time-step correlations.
      Branch B  (3×7) — broken triads: 7-semitone span covers a perfect fifth
                         (the most stable harmonic interval), capturing blocked
                         and arpeggiated tonal structures.
      Branch C  (7×3) — harmonic rhythm: 7 time-step span tracks how harmonic
                         voicings evolve across approximately two beats,
                         capturing phrase-level rhythmic patterns.

    After concatenation, Squeeze-Excitation attention globally reweights the
    c_in*3 multi-scale channels before projection to 1-channel output.

    With the current channel schedule (c_in=12):
      c_cat = 12 * 3 = 36
      SE bottleneck = 36 // 4 = 9
      conv_mix:    36 → 12
      conv_refine: 12 →  6
      conv_out:     6 →  1

    Input  shape: (B, 12, 128, 88)
    Output shape: (B,  1, 128, 88)  values in [-1, 1] via Tanh
    """

    def __init__(self, c_in: int = 16):
        super().__init__()
        c_cat = c_in * 3   # 48 after concatenation

        # ── Three parallel 2D conv branches ───────────────────────────────────
        self.branch_a = nn.Conv2d(c_in, c_in, kernel_size=(3, 3),
                                  padding=(1, 1), bias=False)
        self.branch_b = nn.Conv2d(c_in, c_in, kernel_size=(3, 7),
                                  padding=(1, 3), bias=False)
        self.branch_c = nn.Conv2d(c_in, c_in, kernel_size=(7, 3),
                                  padding=(3, 1), bias=False)

        self.bn1  = nn.BatchNorm2d(c_cat)
        self.act1 = nn.ReLU(inplace=True)

        # ── SE attention on the (c_in*3)-channel concatenation ───────────────────
        self.se   = SqueezeExcitation(c_cat, reduction=4)

        # ── Channel mix 48 → 16 ───────────────────────────────────────────────
        self.conv_mix = nn.Conv2d(c_cat, c_in, kernel_size=1, bias=False)
        self.bn2      = nn.BatchNorm2d(c_in)
        self.act2     = nn.ReLU(inplace=True)

        # ── Local refinement 16 → 8 ───────────────────────────────────────────
        self.conv_refine = nn.Conv2d(c_in, c_in // 2, kernel_size=(3, 3),
                                     padding=(1, 1), bias=False)
        self.bn3         = nn.BatchNorm2d(c_in // 2)
        self.act3        = nn.ReLU(inplace=True)

        # ── Output projection 8 → 1 ───────────────────────────────────────────
        self.conv_out = nn.Conv2d(c_in // 2, 1, kernel_size=1, bias=True)
        self.tanh     = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Multi-scale feature extraction ────────────────────────────────────
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        x = torch.cat([a, b, c], dim=1)   # (B, 48, T, P)
        x = self.act1(self.bn1(x))

        # ── SE channel reweighting ─────────────────────────────────────────────
        x = self.se(x)

        # ── Projection + local refinement ─────────────────────────────────────
        x = self.act2(self.bn2(self.conv_mix(x)))       # (B, 16, T, P)
        x = self.act3(self.bn3(self.conv_refine(x)))    # (B,  8, T, P)

        # ── Output ─────────────────────────────────────────────────────────────
        x = self.tanh(self.conv_out(x))                 # (B,  1, T, P)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# EntryConvBlock — Discriminator entry (mirrors EnhancedFusionBlock)
# ═════════════════════════════════════════════════════════════════════════════

class EntryConvBlock(nn.Module):
    """
    Multi-scale 2D entry block for the discriminator.
    Symmetric counterpart to EnhancedFusionBlock: maps 1 → c_out channels
    using the same three kernel shapes (3×3, 3×7, 7×3), ensuring the
    discriminator sees the same multi-scale inductive bias as the generator’s
    output head.

    The inner branch width is hardcoded to 8 channels per branch (24 total)
    and projected to c_out.  With c_out=12 (current schedule):
      24 channels concatenated → project to 12

    Uses LayerNorm (GroupNorm(1, 24)) — no batch statistics — for WGAN-GP.

    Input  shape: (B,  1, 128, 88)
    Output shape: (B, 12, 128, 88)
    """

    def __init__(self, c_out: int = 16):
        super().__init__()
        # 3 branches × 8 channels = 24 total, then project to c_out (12)
        branch_c    = 8
        c_cat       = branch_c * 3   # 24

        self.branch_a = nn.Conv2d(1, branch_c, kernel_size=(3, 3),
                                  padding=(1, 1), bias=False)
        self.branch_b = nn.Conv2d(1, branch_c, kernel_size=(3, 7),
                                  padding=(1, 3), bias=False)
        self.branch_c = nn.Conv2d(1, branch_c, kernel_size=(7, 3),
                                  padding=(3, 1), bias=False)

        self.norm = nn.GroupNorm(1, c_cat)   
        self.act  = nn.LeakyReLU(0.2, inplace=True)
      
        self.proj = nn.Conv2d(c_cat, c_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        x = torch.cat([a, b, c], dim=1)   
        x = self.act(self.norm(x))
        x = self.proj(x)                  
        return x
