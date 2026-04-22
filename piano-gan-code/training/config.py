from dataclasses import dataclass


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    train_npz:   str = ""          # Path to train.npz
    val_npz:     str = ""          # Path to validation.npz
    test_npz:    str = ""          # Path to test.npz (post-training eval)
    output_dir:  str = "./runs/default"

    z_dim:        int  = 64

    use_temporal: bool = True      
    use_pitch:    bool = True      
    use_fusion:   bool = True      

    # ── Training ───────────────────────────────────────────────────────────
    batch_size:  int   = 32        
    lr_g:        float = 1e-4
    lr_d:        float = 1e-4
    beta1:       float = 0.5       
    beta2:       float = 0.9       
    n_critic:    int   = 3         
    gp_lambda:   float = 10.0      
    max_steps:   int   = 25_000
    val_every:   int   = 1_000     # Compute val metrics every N steps
    save_every:  int   = 500       # Save checkpoint every N steps

    num_workers: int   = 0         # Set to 0 strictly to avoid PyTorch silent RAM OOM crashes on Kaggle.

    # ── WandB ──────────────────────────────────────────────────────────────
    use_wandb:      bool = True
    wandb_project:  str  = "piano-gan"
    wandb_run:      str  = "full"   # "full" | "no-temporal" | "no-pitch" | "baseline-2d"

    # ── Inference / generation ─────────────────────────────────────────────
    n_generate:     int   = 64      
    gen_threshold:  float = 0.5     # Binary threshold after rescaling to [0,1]
    device:         str   = "cuda"

CONFIGS = {
    "full": Config(
        wandb_run="full",
        use_temporal=True, use_pitch=True, use_fusion=True,
    ),
    "no-temporal": Config(
        wandb_run="no-temporal",
        output_dir="./runs/no-temporal",
        use_temporal=False, use_pitch=True, use_fusion=True,
    ),
    "no-pitch": Config(
        wandb_run="no-pitch",
        output_dir="./runs/no-pitch",
        use_temporal=True, use_pitch=False, use_fusion=True,
    ),
    "baseline-2d": Config(
        wandb_run="baseline-2d",
        output_dir="./runs/baseline-2d",
        use_temporal=False, use_pitch=False, use_fusion=True,
    ),
}
