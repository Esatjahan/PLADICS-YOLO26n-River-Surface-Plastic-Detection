# File: D:\PLADICS\scripts\plot\multiclass_bar_chart.py

import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------

datasets = [
    "RiSID 2-Class",
    "RiSID 5-Class",
    "RiSID 7-Class"
]

map50 = [0.815, 0.781, 0.755]

# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(datasets, map50)

# Labels
ax.set_xlabel("Dataset Configuration", fontsize=12)
ax.set_ylabel("mAP@0.5", fontsize=12)
ax.set_title("Multiclass Riverine Plastic Detection Performance", fontsize=14)

# Value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f'{height:.3f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()

# Save
plt.savefig("D:/PLADICS/scripts/plot/multiclass_performance.png", dpi=300)

plt.show()