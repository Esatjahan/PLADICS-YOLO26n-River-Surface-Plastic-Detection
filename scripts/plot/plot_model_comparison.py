import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# Model names
# =========================
models = [
    "YOLO26n",
    "YOLO26n_v2",
    "YOLO26n_img768",
    "YOLO26n_finetune"
]

# =========================
# Verified metrics
# =========================
precision = [0.857, 0.856, 0.779, 0.810]
recall    = [0.772, 0.762, 0.707, 0.725]
map50     = [0.843, 0.837, 0.739, 0.778]
map5095   = [0.527, 0.522, 0.440, 0.454]

# =========================
# Output folder
# =========================
save_dir = r"D:\PLADICS\results\figures"
os.makedirs(save_dir, exist_ok=True)
save_path_png = os.path.join(save_dir, "model_comparison_paper.png")
save_path_pdf = os.path.join(save_dir, "model_comparison_paper.pdf")

# =========================
# Plot settings
# =========================
x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(13, 7))

bars1 = ax.bar(x - 1.5 * width, precision, width, label="Precision")
bars2 = ax.bar(x - 0.5 * width, recall,    width, label="Recall")
bars3 = ax.bar(x + 0.5 * width, map50,     width, label="mAP@50")
bars4 = ax.bar(x + 1.5 * width, map5095,   width, label="mAP@50:95")

# Axis labels and title
ax.set_xlabel("Model Variants", fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Performance Comparison of YOLO26n-Based Model Variants", fontsize=15, pad=14)

# Ticks
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=12, ha="center", fontsize=11)
ax.tick_params(axis='y', labelsize=11)

# Y-axis range
ax.set_ylim(0, 1.0)

# Grid
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
ax.set_axisbelow(True)

# Legend
ax.legend(loc="upper right", fontsize=11, frameon=True)

# Value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.012,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Cleaner borders
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

# Save high-quality outputs
plt.savefig(save_path_png, dpi=600, bbox_inches="tight")
plt.savefig(save_path_pdf, dpi=600, bbox_inches="tight")

print(f"\nFigure saved as PNG: {save_path_png}")
print(f"Figure saved as PDF: {save_path_pdf}")

plt.show()