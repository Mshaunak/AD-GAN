import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Clean, white, publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = '#dddddd'

MODELS = ["midinet", "musegan", "baseline-2d", "full"]
MODEL_NAMES = ["MidiNet", "MuseGAN", "Baseline 2D", "AD-GAN"]
METRICS = ["EBR", "mean_polyphony", "RPS", "PCHS"]
METRIC_LABELS = [r"Empty Bar Rate (EBR) $\downarrow$", "Polyphony", r"RPS $\uparrow$", r"PCHS $\downarrow$"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def load_metrics():
    results = {m: [] for m in METRICS}
    valid_models = []
    valid_names = []
    colors_to_use = []
    
    for idx, model in enumerate(MODELS):
        json_path = None
        out_dir = os.path.join("generated_outputs", model)
        if not os.path.exists(out_dir): continue
        
        for f in os.listdir(out_dir):
            if f.endswith("metrics.json"):
                json_path = os.path.join(out_dir, f)
                break
                
        if json_path:
            with open(json_path, "r") as f:
                data = json.load(f)["aggregate"]
                for m in METRICS:
                    val = data.get(m, 0.0)
                    if isinstance(val, str) and "N/A" in val: val = 0.0
                    results[m].append(float(val))
            valid_models.append(model)
            valid_names.append(MODEL_NAMES[idx])
            colors_to_use.append(COLORS[idx])
            
    return results, valid_names, colors_to_use, valid_models

def plot_bar_charts(results, names, colors):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (ax, metric, label) in enumerate(zip(axes, METRICS, METRIC_LABELS)):
        bars = ax.bar(names, results[metric], color=colors, edgecolor='black', linewidth=0.8)
        ax.set_title(label, fontweight='bold')
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/metrics_comparison.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/metrics_comparison.png", dpi=300, bbox_inches='tight')
    print("[INFO] Saved metrics bar charts to paper/figures/metrics_comparison.png/pdf")

def plot_piano_rolls(models, names):
    # Find one non-silent piano roll per model if possible
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    if len(models) == 1: axes = [axes]
    
    for ax, model, name in zip(axes, models, names):
        out_dir = os.path.join("generated_outputs", model)
        npy_files = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        
        best_roll = None
        best_density = -1
        
        for f in npy_files[:20]:  # check first 20
            roll = np.load(os.path.join(out_dir, f))
            density = roll.sum()
            if density > best_density:
                best_density = density
                best_roll = roll
                
        if best_roll is not None:
            # Binarize for clean plotting
            best_roll = (best_roll > 0.5).astype(float)
            ax.imshow(best_roll.T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
        
        ax.set_ylabel("Pitch")
        ax.set_title(f"{name}", fontsize=11, fontweight='bold', pad=4)
        ax.set_yticks([0, 24, 48, 72, 87])
        ax.set_yticklabels(["A0", "C3", "C5", "C7", "C8"])
        
    axes[-1].set_xlabel("Time Step (16th notes)")
    plt.tight_layout()
    plt.savefig("paper/figures/piano_rolls_comparison.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/piano_rolls_comparison.png", dpi=300, bbox_inches='tight')
    print("[INFO] Saved piano rolls to paper/figures/piano_rolls_comparison.png/pdf")

if __name__ == "__main__":
    results, names, colors, models = load_metrics()
    if names:
        plot_bar_charts(results, names, colors)
        plot_piano_rolls(models, names)
    else:
        print("[ERROR] No metrics found in generated_outputs/")
