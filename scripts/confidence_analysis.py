import os
import matplotlib.pyplot as plt

label_dir = r"D:\PLADICS\confidence_analysis\yolo26n_test_conf\labels"

# Save figure in the same folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "confidence_distribution.png")

confidences = []

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:
                    confidences.append(float(parts[5]))

print("Total detections:", len(confidences))

plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=20)
plt.xlabel("Confidence Score")
plt.ylabel("Number of Detections")
plt.title("YOLO26n Confidence Score Distribution")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved:", save_path)

plt.show()