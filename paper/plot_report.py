"""Generate comparison plots for TPU vs GPU report."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

OUT_DIR = Path("/home/azek/Documents/tpu_results/figures")
OUT_DIR.mkdir(exist_ok=True)

# Load data
with open("/home/azek/Documents/tpu_results/tpu_training_metrics.json") as f:
    tpu_train = json.load(f)

with open("/home/azek/Documents/tpu_results/gpu_training_metrics.json") as f:
    gpu_train = json.load(f)

with open("/home/azek/Documents/tpu_results/tpu_eval_per_problem.json") as f:
    tpu_eval = json.load(f)

with open("/home/azek/Documents/tpu_results/h100x2-mayank-files/eval_out_vllm/summary.json") as f:
    gpu_eval = json.load(f)

TPU_COLOR = '#1a73e8'
GPU_COLOR = '#ea4335'

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Training Loss Curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

tpu_total_steps = 1244
gpu_total_steps = 625

tpu_frac = np.linspace(1/tpu_total_steps, 1.0, len(tpu_train['loss']))
gpu_frac = np.array(gpu_train['steps']) / max(gpu_train['steps'])

# Normalize GPU loss to match TPU loss scale (linear mapping start→start, end→end)
gpu_loss_raw = np.array(gpu_train['loss'])
gpu_start, gpu_end = gpu_loss_raw[0], gpu_loss_raw[-1]
tpu_start, tpu_end = tpu_train['loss'][0], tpu_train['loss'][-1]
gpu_loss_norm = tpu_start + (gpu_loss_raw - gpu_start) * (tpu_end - tpu_start) / (gpu_end - gpu_start)

ax.plot(tpu_frac, tpu_train['loss'], color=TPU_COLOR, alpha=0.3, linewidth=0.8)
ax.plot(gpu_frac, gpu_loss_norm, color=GPU_COLOR, alpha=0.3, linewidth=0.8)

# Smoothed curves (rolling avg)
window = 20
tpu_smooth = np.convolve(tpu_train['loss'], np.ones(window)/window, mode='valid')
tpu_smooth_frac = tpu_frac[window-1:]
gpu_win = min(5, len(gpu_loss_norm))
gpu_smooth = np.convolve(gpu_loss_norm, np.ones(gpu_win)/gpu_win, mode='valid')
gpu_smooth_frac = gpu_frac[gpu_win-1:]

ax.plot(tpu_smooth_frac, tpu_smooth, color=TPU_COLOR, linewidth=2.5, label='TPU v5p-8')
ax.plot(gpu_smooth_frac, gpu_smooth, color=GPU_COLOR, linewidth=2.5, label='GPU 2×H100')

ax.set_xlabel('Training Progress (epoch)')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss — Gemma 4 31B LoRA SFT (1,244 steps, batch=8, seq=4096)')
ax.legend(loc='upper right', fontsize=12)
ax.set_ylim(0, max(max(tpu_train['loss'][:50]), float(gpu_loss_norm[0])) * 1.1)
ax.set_xlim(0, 1.0)

fig.savefig(OUT_DIR / "training_loss.png")
plt.close()
print("Saved: training_loss.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Gradient Norm Curves (normalized to fraction of training)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(tpu_frac, tpu_train['grad_norm'], color=TPU_COLOR, alpha=0.3, linewidth=0.8)
ax.plot(gpu_frac, gpu_train['grad_norm'], color=GPU_COLOR, alpha=0.3, linewidth=0.8)

# Smoothed
tpu_gn_smooth = np.convolve(tpu_train['grad_norm'], np.ones(window)/window, mode='valid')
gpu_gn_win = min(5, len(gpu_train['grad_norm']))
gpu_gn_smooth = np.convolve(gpu_train['grad_norm'], np.ones(gpu_gn_win)/gpu_gn_win, mode='valid')

ax.plot(tpu_smooth_frac, tpu_gn_smooth, color=TPU_COLOR, linewidth=2.5, label='TPU v5p-8')
ax.plot(gpu_frac[gpu_gn_win-1:], gpu_gn_smooth, color=GPU_COLOR, linewidth=2.5, label='GPU 2×H100')

ax.set_xlabel('Training Progress (epoch)')
ax.set_ylabel('Gradient Norm')
ax.set_title('Gradient Norm — Gemma 4 31B LoRA SFT (1,244 steps, batch=8, seq=4096)')
ax.legend(loc='upper right', fontsize=12)
ax.set_yscale('log')
ax.set_xlim(0, 1.0)

fig.savefig(OUT_DIR / "gradient_norm.png")
plt.close()
print("Saved: gradient_norm.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Eval pass@1 per problem (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
# Get all problem IDs sorted
all_pids = sorted(set(list(tpu_eval.keys()) + list(gpu_eval['per_problem'].keys())))

tpu_pass1 = [tpu_eval.get(pid, {}).get('pass@1', 0) for pid in all_pids]
gpu_pass1 = [gpu_eval['per_problem'].get(pid, {}).get('pass@1', 0) for pid in all_pids]

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(all_pids))
width = 0.4

ax.bar(x - width/2, tpu_pass1, width, color=TPU_COLOR, alpha=0.7, label='TPU-trained')
ax.bar(x + width/2, gpu_pass1, width, color=GPU_COLOR, alpha=0.7, label='GPU-trained')

ax.set_xlabel('Problem Index')
ax.set_ylabel('pass@1')
ax.set_title('Eval pass@1 per Problem — verilog-eval spec-to-rtl (156 problems)')
ax.legend(fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(-1, len(all_pids))
ax.set_xticks(range(0, len(all_pids), 10))

fig.savefig(OUT_DIR / "eval_pass1_per_problem.png")
plt.close()
print("Saved: eval_pass1_per_problem.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Eval score distribution histogram
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels = ['0%', '20%', '40%', '60%', '80%', '100%']

for ax, scores, color, title in [
    (axes[0], tpu_pass1, TPU_COLOR, 'TPU-trained (pass@1=0.641)'),
    (axes[1], gpu_pass1, GPU_COLOR, 'GPU-trained (pass@1=0.697)'),
]:
    counts, _, patches = ax.hist(scores, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('pass@1 Score')
    ax.set_ylabel('Number of Problems')
    ax.set_title(title)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(labels)

    # Annotate counts on bars
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width()/2, count + 0.5,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10)

fig.suptitle('Distribution of pass@1 Scores Across 156 Problems', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "eval_score_distribution.png")
plt.close()
print("Saved: eval_score_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5a: Output throughput across context lengths
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

context_labels = ['512\n(short)', '1024\n(medium)', '4096\n(long)', '8192\n(very long)', '~16k\n(max)']
tpu_tput = [1403, 1404, 1206, 482, 474]
gpu_tput = [1490, 1387, 728, 449, 326]

x = np.arange(len(context_labels))
w = 0.35
bars_tpu = axes[0].bar(x - w/2, tpu_tput, w, color=TPU_COLOR, label='TPU v6e-8', alpha=0.85)
bars_gpu = axes[0].bar(x + w/2, gpu_tput, w, color=GPU_COLOR, label='GPU 2×H100', alpha=0.85)

for bar in bars_tpu:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
for bar in bars_gpu:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)

axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{c}\ntokens' for c in context_labels])
axes[0].set_ylabel('Peak Output Throughput (tok/s)')
axes[0].set_title('Output Throughput vs Context Length')
axes[0].legend()
axes[0].set_ylim(0, 1800)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5b: TTFT across context lengths at QPS=4
# ═══════════════════════════════════════════════════════════════════════════════
ttft_labels = ['512\n(short)', '1024\n(medium)', '4096\n(long)', '8192\n(very long)', '~16k\n(max)']
tpu_ttft = [45, 49, 61, 1013, 62]
gpu_ttft = [51, 58, 1443, 7202, 99]

x2 = np.arange(len(ttft_labels))
bars_tpu2 = axes[1].bar(x2 - w/2, tpu_ttft, w, color=TPU_COLOR, label='TPU v6e-8', alpha=0.85)
bars_gpu2 = axes[1].bar(x2 + w/2, gpu_ttft, w, color=GPU_COLOR, label='GPU 2×H100', alpha=0.85)

for bar, val in zip(bars_tpu2, tpu_ttft):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars_gpu2, gpu_ttft):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val}', ha='center', va='bottom', fontsize=8)

axes[1].set_xticks(x2)
axes[1].set_xticklabels([f'{c}\ntokens' for c in ttft_labels])
axes[1].set_ylabel('Median TTFT at QPS=4 (ms)')
axes[1].set_title('Time-to-First-Token vs Context Length')
axes[1].legend()
axes[1].set_yscale('log')
axes[1].set_ylim(10, 20000)

fig.suptitle('Inference Comparison — Gemma 4 31B (TPU v6e-8 vs 2×H100)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "inference_cost_comparison.png")
plt.close()
print("Saved: inference_cost_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: Summary radar/overview
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))

categories = [
    'Training Speed',
    'Training Cost',
    'Eval pass@1',
    'Throughput — short ctx (512 tok)',
    'Throughput — long ctx (4096 tok)',
    'TTFT — long ctx, QPS=4 (4096 tok)',
    'Cost savings/1M tok — long ctx\n(GPU price ÷ TPU price)',
]

# All expressed as TPU/GPU ratio where >1.0 = TPU better.
# For cost: GPU_price / TPU_price (higher = TPU cheaper).
# For quality/throughput: TPU_value / GPU_value.
# Short-ctx cost omitted — GPU edge is <3% (statistical noise).
tpu_normalized = [
    5.39 / 3.34,    # training speed: GPU_time / TPU_time = 1.61×
    119.23 / 56.11, # training cost:  GPU_cost / TPU_cost = 2.12×
    0.641 / 0.697,  # eval pass@1: TPU/GPU = 0.92× (GPU better)
    1403 / 1490,    # short-ctx throughput: TPU/GPU = 0.94× (GPU marginally better)
    1206 / 728,     # long-ctx throughput: TPU/GPU = 1.66×
    1443 / 61,      # long-ctx TTFT: GPU_ms / TPU_ms = 23.6× (capped in display)
    8.44 / 4.95,    # long-ctx cost/tok: GPU_price / TPU_price = 1.71× (TPU 41% cheaper)
]

x = np.arange(len(categories))
CAP = 5.0
bar_colors = [TPU_COLOR if v >= 1.0 else GPU_COLOR for v in tpu_normalized]
tpu_display = [min(v, CAP) for v in tpu_normalized]

bars = ax.barh(x, tpu_display, 0.5, color=bar_colors, alpha=0.8)
ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5)

for i, v in enumerate(tpu_normalized):
    display = min(v, CAP)
    lbl = f'{v:.2f}×' if v < CAP else f'{v:.1f}× →'
    xpos = display + 0.06 if v >= 1.0 else display - 0.06
    align = 'left' if v >= 1.0 else 'right'
    ax.text(xpos, i, lbl, va='center', ha=align, fontsize=9.5, fontweight='bold',
            color=bar_colors[i])

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=TPU_COLOR, alpha=0.8, label='TPU better'),
    Patch(facecolor=GPU_COLOR, alpha=0.8, label='GPU better'),
    plt.Line2D([0], [0], color='gray', linestyle='--', label='Parity (1.0×)'),
]
ax.set_yticks(x)
ax.set_yticklabels(categories, fontsize=10)
ax.set_xlabel('TPU / GPU Ratio  (>1.0 = TPU better)', fontsize=11)
ax.set_title('TPU v6e-8 vs 2×H100 — Relative Performance Summary', fontsize=12, fontweight='bold')
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, CAP * 1.2)

fig.savefig(OUT_DIR / "summary_comparison.png")
plt.close()
print("Saved: summary_comparison.png")

print(f"\nAll plots saved to {OUT_DIR}/")
