# File: D:\PLADICS\scripts\plot\benchmark_bar_chart.py

import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Corrected clean-dataset results
# -----------------------------

models = [
    "YOLOv10n",
    "YOLOv8n",
    "YOLO11n",
    "YOLO26n",
    "YOLO26s",
    "YOLO26n-100"
]

map50 = [0.693, 0.868, 0.767, 0.866, 0.876, 0.915]
map5095 = [0.431, 0.562, 0.482, 0.556, 0.580, 0.608]

x = np.arange(len(models))
width = 0.34

# -----------------------------
# Publication-style settings
# -----------------------------

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(8.2, 4.6))

bars1 = ax.bar(
    x - width / 2,
    map50,
    width,
    label="mAP@0.5",
    edgecolor="black",
    linewidth=0.5
)

bars2 = ax.bar(
    x + width / 2,
    map5095,
    width,
    label="mAP@0.5:0.95",
    edgecolor="black",
    linewidth=0.5
)

# -----------------------------
# Axes formatting
# -----------------------------

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Detection Score", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)

ax.set_ylim(0.35, 0.98)
ax.set_yticks(np.arange(0.4, 1.0, 0.1))
ax.tick_params(axis="y", labelsize=10)

ax.legend(
    frameon=False,
    fontsize=10,
    loc="upper left",
    ncol=2
)

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -----------------------------
# Value labels
# -----------------------------

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------

output_dir = r"D:\PLADICS\figures"
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, "benchmark_comparison.png")
pdf_path = os.path.join(output_dir, "benchmark_comparison.pdf")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")