"""
training/train.py — Main WGAN-GP training loop for the Axis-Decomposed Piano GAN.

Usage (local / Kaggle notebook):
  from training.config import Config
  from training.train import train

  cfg = Config(
      train_npz   = "/kaggle/input/piano-rolls-npz/train.npz",
      val_npz     = "/kaggle/input/piano-rolls-npz/validation.npz",
      output_dir  = "/kaggle/working/runs/full",
      wandb_run   = "full",
  )
  train(cfg)

Multi-GPU: wraps both G and D in nn.DataParallel if >1 CUDA device is
  available (Kaggle T4×2).  No code changes needed between 1-GPU and 2-GPU.

Checkpointing: saves every save_every steps to output_dir/ckpt_step{N}.pt.
  On restart, automatically loads the latest checkpoint and resumes.
"""

import os
import re
import glob
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset       import PianoRollDataset
from models.generator   import Generator
from models.discriminator import Discriminator
from training.config    import Config
from training.losses    import generator_loss, discriminator_loss, gradient_penalty
from evaluation.metrics import empty_bar_rate, avg_polyphony

scaler = torch.cuda.amp.GradScaler()

log = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(output_dir: str, G, D, opt_G, opt_D, step: int):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ckpt_step{step:07d}.pt")
    torch.save({
        "step":  step,
        "G":     G.state_dict(),
        "D":     D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
    }, path)
    log.info(f"Checkpoint saved → {path}")


def _load_latest_checkpoint(output_dir: str, G, D, opt_G, opt_D) -> int:
    """Load the most recent checkpoint in output_dir. Returns resumed step (0 if none)."""
    ckpts = glob.glob(os.path.join(output_dir, "ckpt_step*.pt"))
    if not ckpts:
        log.info("No checkpoint found — training from scratch.")
        return 0

    # Pick checkpoint with highest step number
    def _step(path):
        m = re.search(r"step(\d+)", path)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_step)
    ckpt   = torch.load(latest, map_location="cpu")

    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])

    step = ckpt["step"]
    log.info(f"Resumed from {latest}  (step {step:,})")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Fixed noise for qualitative inspection
# ─────────────────────────────────────────────────────────────────────────────

def _make_fixed_noise(n: int, z_dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic noise for consistent visual logging across steps."""
    rng = torch.Generator()   # always CPU
    rng.manual_seed(42)
    return torch.randn(n, z_dim, generator=rng).to(device)  # generate CPU, move to GPU


# ─────────────────────────────────────────────────────────────────────────────
# Data iterator that never runs out
# ─────────────────────────────────────────────────────────────────────────────

def _infinite_loader(loader: DataLoader):
    """Cycle over a DataLoader indefinitely."""
    while True:
        yield from loader


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    """
    End-to-end WGAN-GP training loop.

    All hyperparameters come from `cfg` (training.config.Config).
    Call this function from a script or Kaggle notebook cell.
    """

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator(
        z_dim        = cfg.z_dim,
        use_temporal = cfg.use_temporal,
        use_pitch    = cfg.use_pitch,
        use_fusion   = cfg.use_fusion,
    ).to(device)

    D = Discriminator(
        use_temporal = cfg.use_temporal,
        use_pitch    = cfg.use_pitch,
    ).to(device)

    # Multi-GPU (Kaggle T4×2)
    if torch.cuda.device_count() > 1:
        log.info(f"Using nn.DataParallel across {torch.cuda.device_count()} GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    G_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    D_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    log.info(f"G params: {G_params:,}  |  D params: {D_params:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(
        G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
    )
    opt_D = torch.optim.Adam(
        D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PianoRollDataset(cfg.train_npz, augment=True)
    val_ds   = PianoRollDataset(cfg.val_npz,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = False,   # Strictly False to avoid known PyTorch background buffer leaks
        drop_last   = True,    # ensures every batch is full (important for GP)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = False,
    )

    log.info(f"Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}")

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Persistent disk log (appends across restarts) ─────────────────────────
    log_path    = os.path.join(cfg.output_dir, "train.log")
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(file_handler)
    log.info(f"Disk log → {log_path}")

    start_step = _load_latest_checkpoint(cfg.output_dir, G, D, opt_G, opt_D)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb = None
    if cfg.use_wandb:
        try:
            import wandb as _wandb  # type: ignore
        except Exception as e:
            log.warning(f"WandB disabled (import failed): {e}")
            cfg.use_wandb = False
        else:
            wandb = _wandb
            wandb.init(
                project = cfg.wandb_project,
                name    = cfg.wandb_run,
                config  = vars(cfg),
                resume  = "allow",
            )
            # Avoid `log="all"` (parameter histograms) to reduce memory overhead on Kaggle.
            wandb.watch(G, log="gradients", log_freq=1000)

    # ── Fixed noise for consistent visual logging ──────────────────────────────
    fixed_z = _make_fixed_noise(16, cfg.z_dim, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    data_iter = _infinite_loader(train_loader)
    G.train()
    D.train()

    for step in range(start_step, cfg.max_steps):
        # (Type checker appeasement) These are always set because cfg.n_critic >= 1,
        # but initializing avoids "possibly unbound" diagnostics.
        loss_d_wgan = None
        gp = None
        real_scores = None
        fake_scores = None

        # =======================
        # Get real batch
        # =======================
        real = next(data_iter).to(device, non_blocking=True)
        B = real.size(0)

        # =======================
        # DISCRIMINATOR (n_critic steps)
        # =======================
        for _ in range(cfg.n_critic):

            z = torch.randn(B, cfg.z_dim, device=device)

            D_raw = D.module if isinstance(D, nn.DataParallel) else D

            with torch.cuda.amp.autocast():
                fake = G(z)

                real_scores = D(real)
                fake_scores = D(fake.detach())

                loss_d_wgan = discriminator_loss(real_scores, fake_scores)

            # Compute Gradient Penalty outside autocast (in fp32) to prevent NaN/overflow
            gp = gradient_penalty(
                D_raw, real, fake.detach(), device, cfg.gp_lambda
            )

            loss_D = loss_d_wgan + gp

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            # Required: GradScaler only allows one `step()` per `update()`.
            # Since we step D multiple times (n_critic), we must update each time.
            scaler.update()

        # =======================
        # GENERATOR (1 step)
        # =======================
        z = torch.randn(B, cfg.z_dim, device=device)

        with torch.cuda.amp.autocast():
            fake = G(z)
            g_scores = D(fake)
            loss_G = generator_loss(g_scores)

        opt_G.zero_grad()
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        scaler.update()

        # =======================
        # LOGGING
        # =======================
        if step % cfg.log_every == 0:
            assert loss_d_wgan is not None
            assert gp is not None
            assert real_scores is not None
            assert fake_scores is not None
            log_dict = {
                "loss_G": loss_G.item(),
                "loss_D": loss_d_wgan.item(),
                "grad_penalty": gp.item(),
                "D_real": real_scores.mean().item(),
                "D_fake": fake_scores.mean().item(),
                "step": step,
            }
            log.info(
                f"Step {step:6d}  "
                f"lossG={log_dict['loss_G']:.4f}  "
                f"lossD={log_dict['loss_D']:.4f}  "
                f"GP={log_dict['grad_penalty']:.4f}  "
                f"D_real={log_dict['D_real']:.3f}  "
                f"D_fake={log_dict['D_fake']:.3f}"
            )
            if cfg.use_wandb and wandb is not None:
                wandb.log(log_dict, step=step)

        # =======================
        # VALIDATION
        # =======================
        if step % cfg.val_every == 0:
            G.eval()
            with torch.no_grad():
                z_val = torch.randn(cfg.n_generate, cfg.z_dim, device=device)
                samples = G(z_val)
                samples = (samples + 1.0) / 2.0
                samples_bin = (samples > cfg.gen_threshold).float()

                ebr = empty_bar_rate(samples_bin)
                poly = avg_polyphony(samples_bin)

            log.info(f"  ↳ [val] EBR={ebr:.4f}  Polyphony={poly:.3f}")
            if cfg.use_wandb and wandb is not None:
                wandb.log({"val_EBR": ebr, "val_Polyphony": poly}, step=step)
            G.train()

        # =======================
        # CHECKPOINT
        # =======================
        if step > 0 and step % cfg.save_every == 0:
            _save_checkpoint(cfg.output_dir, G, D, opt_G, opt_D, step)
            if cfg.use_wandb and wandb is not None:
                ckpt_path = os.path.join(cfg.output_dir, f"ckpt_step{step:07d}.pt")
                artifact = wandb.Artifact(f"checkpoint-{cfg.wandb_run}", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
                log.info(f"Uploaded checkpoint to WandB Artifacts")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    _save_checkpoint(cfg.output_dir, G, D, opt_G, opt_D, cfg.max_steps)
    log.info("Training complete.")

    if cfg.use_wandb and wandb is not None:
        wandb.finish()

    # ── Remove FileHandler to avoid duplicates if train() is called again ──────
    log.removeHandler(file_handler)
    file_handler.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (alternative to notebook usage)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from training.config import CONFIGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        default="full",
                        choices=list(CONFIGS.keys()),
                        help="Which ablation run to execute")
    parser.add_argument("--train_npz",  required=True)
    parser.add_argument("--val_npz",    required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.run]
    cfg.train_npz  = args.train_npz
    cfg.val_npz    = args.val_npz
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_wandb:
        cfg.use_wandb  = False

    train(cfg)
