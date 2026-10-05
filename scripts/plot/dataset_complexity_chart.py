# File: D:\PLADICS\scripts\plot\dataset_complexity_chart.py

import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------

datasets = [
    "Single-Class",
    "RiSID 2-Class",
    "RiSID 5-Class",
    "RiSID 7-Class"
]

map50 = [
    0.915,   # Single-class (YOLO26n-100)
    0.815,   # RiSID 2-class
    0.781,   # RiSID 5-class
    0.755    # RiSID 7-class
]

# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(datasets, map50)

# Labels
ax.set_xlabel("Dataset Configuration", fontsize=12)
ax.set_ylabel("mAP@0.5", fontsize=12)

ax.set_title(
    "Impact of Dataset Complexity on Detection Performance",
    fontsize=14
)

# Y-axis range
ax.set_ylim(0, 1.0)

# Value labels
for bar in bars:
    height = bar.get_height()

    ax.annotate(
        f'{height:.3f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()

# Save
plt.savefig(
    "D:/PLADICS/scripts/plot/dataset_complexity_chart.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()