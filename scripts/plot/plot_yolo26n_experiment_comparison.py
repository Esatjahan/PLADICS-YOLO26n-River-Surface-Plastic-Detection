import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# MANUAL FINAL METRICS
# =========================
data = [
    {
        "Experiment": "Merged Single-Class",
        "Precision": 0.8341,
        "Recall": 0.7382,
        "mAP50": 0.7896,
        "mAP50_95": 0.4582,
    },
    {
        "Experiment": "RiSID Multiclass",
        "Precision": 0.6840,
        "Recall": 0.6880,
        "mAP50": 0.7320,
        "mAP50_95": 0.4820,
    },
    {
        "Experiment": "Riverine Multiclass",
        "Precision": 0.5640,
        "Recall": 0.6590,
        "mAP50": 0.7140,
        "mAP50_95": 0.3960,
    },
]

df = pd.DataFrame(data)

# =========================
# OUTPUT PATHS
# =========================
fig_dir = r"D:\PLADICS\results\figures"
table_dir = r"D:\PLADICS\results\tables"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

csv_path = os.path.join(table_dir, "yolo26n_experiment_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"[OK] Saved CSV: {csv_path}")

# =========================
# STYLE
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

metrics = ["Precision", "Recall", "mAP50", "mAP50_95"]

# =========================
# INDIVIDUAL PUBLICATION-STYLE PLOTS
# =========================
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(df["Experiment"], df[metric])

    ax.set_title(f"{metric} Comparison Across YOLO26n Experiment Settings", pad=12)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=15, ha="right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    save_png = os.path.join(fig_dir, f"yolo26n_{metric.lower()}_comparison.png")
    save_pdf = os.path.join(fig_dir, f"yolo26n_{metric.lower()}_comparison.pdf")
    plt.savefig(save_png, dpi=600, bbox_inches="tight")
    plt.savefig(save_pdf, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {save_png}")
    print(f"[OK] Saved: {save_pdf}")

# =========================
# COMBINED GROUPED BAR CHART
# =========================
fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(df))
width = 0.18

bars1 = ax.bar(x - 1.5 * width, df["Precision"], width=width, label="Precision")
bars2 = ax.bar(x - 0.5 * width, df["Recall"], width=width, label="Recall")
bars3 = ax.bar(x + 0.5 * width, df["mAP50"], width=width, label="mAP50")
bars4 = ax.bar(x + 1.5 * width, df["mAP50_95"], width=width, label="mAP50-95")

ax.set_xticks(x)
ax.set_xticklabels(df["Experiment"], rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("YOLO26n Performance Across Different Dataset Settings", pad=12)
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout()

combined_png = os.path.join(fig_dir, "yolo26n_combined_experiment_comparison.png")
combined_pdf = os.path.join(fig_dir, "yolo26n_combined_experiment_comparison.pdf")
plt.savefig(combined_png, dpi=600, bbox_inches="tight")
plt.savefig(combined_pdf, dpi=600, bbox_inches="tight")
plt.close()

print(f"[OK] Saved: {combined_png}")
print(f"[OK] Saved: {combined_pdf}")