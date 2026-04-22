"""
training/losses.py — WGAN-GP loss functions.

WGAN-GP (Wasserstein GAN with Gradient Penalty) replaces the original WGAN
weight-clipping with a differentiable gradient penalty that enforces the
1-Lipschitz constraint directly.

Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
           https://arxiv.org/abs/1704.00028

Key equations
─────────────
  L_D = E[D(G(z))] − E[D(x_real)] + λ · GP
  L_G = −E[D(G(z))]
  GP  = E[(‖∇_x̂ D(x̂)‖₂ − 1)²]
        where x̂ = α·x_real + (1−α)·G(z),  α ~ Uniform(0,1)
"""

import torch
import torch.autograd as autograd


# ─────────────────────────────────────────────────────────────────────────────
# Generator loss
# ─────────────────────────────────────────────────────────────────────────────

def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    WGAN-GP generator loss: maximise D(fake), minimise its negative.

    Parameters
    ----------
    fake_scores : (B, 1)  Discriminator scores for generated samples.

    Returns
    -------
    scalar loss tensor — call .backward() on this.
    """
    return -fake_scores.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator loss (critic loss, without gradient penalty)
# ─────────────────────────────────────────────────────────────────────────────

def discriminator_loss(real_scores: torch.Tensor,
                       fake_scores: torch.Tensor) -> torch.Tensor:
    """
    WGAN-GP critic loss: maximise D(real) − D(fake)  ≡  minimise D(fake) − D(real).

    Parameters
    ----------
    real_scores : (B, 1)  Discriminator scores for real samples.
    fake_scores : (B, 1)  Discriminator scores for generated samples.

    Returns
    -------
    scalar loss tensor (add gradient_penalty() before calling .backward()).
    """
    return fake_scores.mean() - real_scores.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient penalty
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(D: torch.nn.Module,
                     real:    torch.Tensor,
                     fake:    torch.Tensor,
                     device:  torch.device,
                     lambda_: float = 10.0) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.

    Interpolates between real and fake samples, runs the discriminator on the
    interpolation, computes ‖∇D‖₂ per sample and penalises deviations from 1.

    Parameters
    ----------
    D       : Discriminator module.
    real    : (B, 1, T, P)  real piano rolls in [-1, 1].
    fake    : (B, 1, T, P)  generated piano rolls  (detached from G graph).
    device  : torch.device
    lambda_ : float  Penalty coefficient (paper uses 10).

    Returns
    -------
    gp : scalar tensor =  lambda_ · E[(‖∇D(x̂)‖₂ − 1)²]

    Notes
    -----
    • fake must NOT require grad when passed in (call .detach() first).
    • We use create_graph=True so that the penalty itself is differentiable
      w.r.t. D's parameters, which is required for the backward pass on L_D.
    • DO NOT use BatchNorm in D (see discriminator.py for rationale).
    """
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)          # per-sample mixing ratio
    alpha = alpha.expand_as(real)

    interp        = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    d_interp      = D(interp)                               # (B, 1)

    grad_outputs  = torch.ones_like(d_interp)
    gradients     = autograd.grad(
        outputs      = d_interp,
        inputs       = interp,
        grad_outputs = grad_outputs,
        create_graph = True,
    )[0]                                                    # (B, 1, T, P)

    gradients     = gradients.view(B, -1)                  # (B, T·P)
    grad_norm     = gradients.norm(2, dim=1)               # (B,)
    gp            = lambda_ * ((grad_norm - 1.0) ** 2).mean()
    return gp
