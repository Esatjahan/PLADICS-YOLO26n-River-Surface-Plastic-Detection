import matplotlib.pyplot as plt

# =========================
# Robustness Data
# =========================

training_data = [100, 75, 50, 25]

map50 = [0.866, 0.844, 0.742, 0.632]

map5095 = [0.556, 0.532, 0.460, 0.379]

# =========================
# Figure Setup
# =========================

plt.figure(figsize=(7.2, 5.2))

# =========================
# Plot Lines
# =========================

plt.plot(
    training_data,
    map50,
    marker='o',
    linewidth=2.2,
    markersize=7,
    label='mAP@0.5'
)

plt.plot(
    training_data,
    map5095,
    marker='s',
    linewidth=2.2,
    markersize=7,
    label='mAP@0.5:0.95'
)

# =========================
# Axis Labels
# =========================

plt.xlabel(
    'Training Data Used (%)',
    fontsize=12
)

plt.ylabel(
    'Detection Performance',
    fontsize=12
)

# =========================
# Ticks and Limits
# =========================

plt.xticks(
    [25, 50, 75, 100],
    fontsize=11
)

plt.yticks(fontsize=11)

plt.ylim(0.3, 0.95)

# =========================
# Grid
# =========================

plt.grid(
    True,
    linestyle='--',
    linewidth=0.7,
    alpha=0.5
)

# =========================
# Legend
# =========================

plt.legend(
    fontsize=11,
    frameon=True
)

# =========================
# Invert Axis
# =========================

plt.gca().invert_xaxis()

# =========================
# Tight Layout
# =========================

plt.tight_layout()

# =========================
# Save Figure
# =========================

plt.savefig(
    r"D:/PLADICS/scripts/robustness/robustness_curve.png",
    dpi=600,
    bbox_inches='tight'
)

print("Robustness curve saved successfully.")

plt.show()