# File: D:\PLADICS\scripts\plot\transfer_learning_bar_chart.py

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------

metrics = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]

s1 = [0.290, 0.401, 0.303, 0.185]
s2 = [0.724, 0.727, 0.781, 0.527]

x = np.arange(len(metrics))
width = 0.35

# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, s1, width, label='S1 (Scratch)')
bars2 = ax.bar(x + width/2, s2, width, label='S2 (Pretrained)')

# Labels
ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Transfer Learning Analysis: Scratch vs Pretrained YOLO26n", fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.legend()

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()

# Save
plt.savefig("D:/PLADICS/scripts/plot/transfer_learning_comparison.png", dpi=300)

plt.show()