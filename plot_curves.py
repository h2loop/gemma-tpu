import json
import matplotlib.pyplot as plt
import numpy as np

with open("training_run_2048.json") as f:
    data = json.load(f)

train_loss = data["training"]["loss"]
train_ppl = data["training"]["perplexity"]
eval_loss = data["eval"]["loss"]
eval_ppl = data["eval"]["perplexity"]

max_steps = data["config"]["max_steps"]
eval_every = data["config"]["eval_every_n_steps"]

train_steps = np.arange(1, max_steps + 1)
eval_steps = np.arange(0, max_steps + 1, eval_every)
if len(eval_steps) != len(eval_loss):
    eval_steps = np.linspace(0, max_steps, len(eval_loss))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Gemma-4 31B LoRA Fine-tuning  |  seq_len=2048", fontsize=13, fontweight="bold")

# Loss
ax = axes[0]
ax.plot(train_steps, train_loss, color="#4C72B0", linewidth=1.2, alpha=0.5, label="Train (per step)")
window = 10
smoothed = np.convolve(train_loss, np.ones(window) / window, mode="valid")
ax.plot(np.arange(window, max_steps + 1), smoothed, color="#4C72B0", linewidth=2.2, label=f"Train (smoothed, w={window})")
ax.plot(eval_steps, eval_loss, "o-", color="#DD8452", linewidth=2, markersize=6, label="Eval")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Perplexity
ax = axes[1]
ax.plot(train_steps, train_ppl, color="#4C72B0", linewidth=1.2, alpha=0.5, label="Train (per step)")
smoothed_ppl = np.convolve(train_ppl, np.ones(window) / window, mode="valid")
ax.plot(np.arange(window, max_steps + 1), smoothed_ppl, color="#4C72B0", linewidth=2.2, label=f"Train (smoothed, w={window})")
ax.plot(eval_steps, eval_ppl, "o-", color="#DD8452", linewidth=2, markersize=6, label="Eval")
ax.set_xlabel("Step")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_2048.png", dpi=150, bbox_inches="tight")
print("Saved training_curves_2048.png")
plt.show()
