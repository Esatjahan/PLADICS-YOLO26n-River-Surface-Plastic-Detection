import pandas as pd
import matplotlib.pyplot as plt
import os

# File path
csv_path = r"D:\PLADICS\results\tables\threshold_sweep.csv"
save_path = r"D:\PLADICS\results\figures\threshold_sweep_plot.png"

# Load CSV
df = pd.read_csv(csv_path)

# Create figure
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(df["conf"], df["precision"], marker="o", label="Precision")
plt.plot(df["conf"], df["recall"], marker="s", label="Recall")
plt.plot(df["conf"], df["f1"], marker="^", label="F1 Score")
plt.plot(df["conf"], df["mAP50"], marker="d", label="mAP@0.5")
plt.plot(df["conf"], df["mAP50_95"], marker="x", label="mAP@0.5:0.95")

# Labels and title
plt.xlabel("Confidence Threshold")
plt.ylabel("Metric Value")
plt.title("Threshold Sweep Performance of YOLO26n_v2")
plt.legend()
plt.grid(True)

# Save figure
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {save_path}")