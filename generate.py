import os
import sys
import json
import argparse
import numpy as np
import torch

try:
    import pretty_midi
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False
    print("[WARN] pretty_midi not installed — MIDI export disabled. Install with: pip install pretty_midi")


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib not installed — PNG export disabled.")


def load_generator(model_name: str, ckpt_path: str, device: torch.device, is_baseline: bool = False):
    """Load the Generator from a checkpoint. Works with partial/crashed checkpoints."""

    # Dynamically import the correct Generator class
    if model_name == "piano-gan":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "piano-gan-code"))
        from models.generator import Generator
        if is_baseline:
            G = Generator(z_dim=64, use_temporal=False, use_pitch=False, use_fusion=True)
        else:
            G = Generator(z_dim=64, use_temporal=True, use_pitch=True, use_fusion=True)
    elif model_name == "midinet":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "midinet-code"))
        from models.generator import Generator
        G = Generator(z_dim=64)
    elif model_name == "musegan":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "musegan-code"))
        from models.generator import Generator
        G = Generator(z_dim=64)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: piano-gan, midinet, musegan")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("G", ckpt)  # supports both full ckpt dicts and raw state dicts

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    G.load_state_dict(state, strict=True)
    G.eval()
    G.to(device)

    step = ckpt.get("step", "unknown")
    print(f"[INFO] Loaded {model_name} Generator from step {step} | {ckpt_path}")
    return G, step


# ─────────────────────────────────────────────────────────────────────────────
# Piano roll → MIDI
# ─────────────────────────────────────────────────────────────────────────────

def roll_to_midi(roll_np: np.ndarray, out_path: str, tempo: float = 120.0, time_per_step: float = 0.0625):
    """
    Convert a binary (T, P) piano roll to a MIDI file.

    roll_np  : (T, P) numpy array, values in {0, 1}
    tempo    : BPM (default 120)
    time_per_step: seconds per grid step. At 120 BPM, 16th note = 0.125s.
                   Default 0.0625 = 32nd notes (64 steps = 2 bars at 120BPM)
    """
    if not HAS_MIDI:
        print(f"[SKIP] MIDI export skipped (pretty_midi not available)")
        return

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    T, P = roll_np.shape
    # A0 on an 88-key piano = MIDI note 21
    MIDI_OFFSET = 21

    for p in range(P):
        in_note = False
        note_start = 0.0
        for t in range(T):
            active = roll_np[t, p] > 0.5
            if active and not in_note:
                note_start = t * time_per_step
                in_note = True
            elif not active and in_note:
                note_end = t * time_per_step
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=p + MIDI_OFFSET,
                    start=note_start,
                    end=note_end,
                )
                piano.notes.append(note)
                in_note = False
        if in_note:
            note = pretty_midi.Note(
                velocity=80,
                pitch=p + MIDI_OFFSET,
                start=note_start,
                end=T * time_per_step,
            )
            piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(out_path)
    print(f"  [MIDI] -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_class_histogram(rolls_bin: np.ndarray) -> np.ndarray:
    P = rolls_bin.shape[-1]
    hist = np.zeros(12, dtype=np.float64)
    for i in range(P):
        pc = (i + 9) % 12  # A0 (index 0) = pitch class 9 = 'A'
        hist[pc] += rolls_bin[..., i].sum()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def compute_pchs(real_bin: np.ndarray, fake_bin: np.ndarray) -> float:
    """
    Pitch Class Histogram Similarity: chi-square distance between
    12-bin pitch-class histograms of real vs generated. Lower = better.
    """
    real_h = _pitch_class_histogram(real_bin)
    fake_h = _pitch_class_histogram(fake_bin)
    chi2 = ((real_h - fake_h) ** 2 / (real_h + fake_h + 1e-8)).sum()
    return float(chi2)


def compute_metrics(rolls_bin: np.ndarray) -> dict:
    """
    rolls_bin: (N, T, P) binary numpy array.
    Returns dict with EBR, mean_polyphony, RPS.
    """
    N, T, P = rolls_bin.shape
    ebr_list, poly_list = [], []

    for i in range(N):
        r = rolls_bin[i]  # (T, P)
        if r.sum() == 0:
            ebr_list.append(1.0)
            poly_list.append(0.0)
        else:
            ebr_list.append(0.0)
            active_steps = r.sum(axis=-1)  # (T,)
            nonzero = active_steps[active_steps > 0]
            poly_list.append(float(nonzero.mean()) if len(nonzero) > 0 else 0.0)

    # Rhythmic Pattern Score
    rps_scores = []
    for i in range(N):
        r = rolls_bin[i]
        onsets = ((r[1:, :] > 0) & (r[:-1, :] == 0)).astype(float)
        density = onsets.sum(axis=-1)  # (T-1,)
        lags = [4, 8, 16]
        corrs = []
        for lag in lags:
            if len(density) <= lag:
                continue
            a = density[:-lag]
            b = density[lag:]
            num = np.dot(a, b)
            den = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-8
            corrs.append(num / den)
        rps_scores.append(float(np.mean(corrs)) if corrs else 0.0)

    return {
        "EBR":          float(np.mean(ebr_list)),
        "mean_polyphony": float(np.mean(poly_list)),
        "RPS":          float(np.mean(rps_scores)),
        "n_samples":    N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate(model_name: str, ckpt_path: str, out_dir: str, n: int = 16, z_dim: int = 64,
             threshold: float = 0.5, seed: int = 42, real_npz: str = None, is_baseline: bool = False):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    G, step = load_generator(model_name, ckpt_path, device, is_baseline=is_baseline)

    # Deterministic noise
    torch.manual_seed(seed)
    z = torch.randn(n, z_dim, device=device)

    with torch.no_grad():
        out = G(z)  # (N, 1, T, P)

    # Rescale from [-1,1] -> [0,1] then threshold
    out_01  = (out.squeeze(1).cpu().numpy() + 1.0) / 2.0  # (N, T, P)
    out_bin = (out_01 > threshold).astype(np.float32)       # (N, T, P)

    print(f"[INFO] Generated {n} samples | shape {out_bin.shape}")

    # Load real data for PCHS if provided
    real_bin = None
    if real_npz:
        print(f"[INFO] Loading real data for PCHS from {real_npz}")
        data = np.load(real_npz)
        key = list(data.keys())[0]
        real_rolls = data[key]  # (N_real, 1, T, P) or (N_real, T, P)
        if real_rolls.ndim == 4:
            real_rolls = real_rolls[:, 0]  # squeeze channel dim
        # Rescale to [0,1] and binarize
        real_01 = (real_rolls + 1.0) / 2.0
        real_bin = (real_01 > threshold).astype(np.float32)
        print(f"[INFO] Real samples loaded: {real_bin.shape}")

    all_metrics = []
    for i in range(n):
        roll_f = out_01[i]   # (T, P) float
        roll_b = out_bin[i]  # (T, P) binary

        # Save raw numpy
        npy_path = os.path.join(out_dir, f"{model_name}_step{step}_sample{i:03d}.npy")
        np.save(npy_path, roll_f)

        # Save PNG
        if HAS_PLT:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(roll_b.T, aspect="auto", origin="lower",
                      cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
            ax.set_xlabel("Time step (16th notes)")
            ax.set_ylabel("Pitch (0=A0, 87=C8)")
            ax.set_title(f"{model_name.upper()} | step {step} | sample {i}")
            png_path = os.path.join(out_dir, f"{model_name}_step{step}_sample{i:03d}.png")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"  [PNG]  -> {png_path}")

        # Save MIDI
        mid_path = os.path.join(out_dir, f"{model_name}_step{step}_sample{i:03d}.mid")
        roll_to_midi(roll_b, mid_path)

        # Per-sample metrics
        m = compute_metrics(roll_b[np.newaxis]) 
        m["sample"] = i
        all_metrics.append(m)

    # Aggregate metrics
    aggregate = compute_metrics(out_bin)
    aggregate["model"]       = model_name
    aggregate["ckpt_step"]   = str(step)
    aggregate["n_generated"] = n
    aggregate["threshold"]   = threshold

    # PCHS (requires real data)
    if real_bin is not None:
        pchs = compute_pchs(real_bin, out_bin)
        aggregate["PCHS"] = pchs
        print(f"  PCHS           : {pchs:.4f}   (lower = closer to real tonal distribution)")
    else:
        aggregate["PCHS"] = "N/A (no --real_npz provided)"

    metrics_path = os.path.join(out_dir, f"{model_name}_step{step}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_sample": all_metrics}, f, indent=2)

    print(f"\n[DONE] Metrics saved -> {metrics_path}")
    print(f"  EBR            : {aggregate['EBR']:.4f}   (lower = better; 0 = no silent bars)")
    print(f"  Mean Polyphony : {aggregate['mean_polyphony']:.3f}   (target: 2–6 for piano)")
    print(f"  RPS            : {aggregate['RPS']:.4f}   (higher = more rhythmic regularity)")

    return aggregate


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate piano rolls from a trained GAN checkpoint.")
    parser.add_argument("--model", required=True, choices=["piano-gan", "midinet", "musegan"],
                        help="Which model architecture to use")
    parser.add_argument("--ckpt",  required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--out",   default="./generated_outputs",
                        help="Output directory for rolls, PNGs, MIDIs, and metrics.json")
    parser.add_argument("--n",     type=int, default=16,
                        help="Number of samples to generate (default: 16)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarisation threshold (default: 0.5)")
    parser.add_argument("--real_npz", default=None,
                        help="Path to real data .npz (e.g. validation.npz) for PCHS metric")
    parser.add_argument("--baseline_2d", action="store_true",
                        help="Set if this is the piano-gan baseline model (no pitch/temporal branches)")
    args = parser.parse_args()

    generate(
        model_name  = args.model,
        ckpt_path   = args.ckpt,
        out_dir     = args.out,
        n           = args.n,
        seed        = args.seed,
        threshold   = args.threshold,
        real_npz    = args.real_npz,
        is_baseline = args.baseline_2d,
    )
