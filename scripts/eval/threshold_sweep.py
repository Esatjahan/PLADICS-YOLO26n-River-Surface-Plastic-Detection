from ultralytics import YOLO

# =========================
# Model and dataset paths
# =========================
model_path = r"D:\PLADICS\experiments_precision\yolo26n_v2\weights\best.pt"
data_path = r"D:\PLADICS\merged_dataset_v2\plastic.yaml"

# Load model
model = YOLO(model_path)

# Confidence thresholds to test
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

results_list = []

print("\n===== Threshold Sweep Started =====\n")

for conf in thresholds:
    print(f"\nRunning for conf = {conf}")

    metrics = model.val(
        data=data_path,
        conf=conf,
        verbose=False
    )

    # Safely extract scalar values
    precision = metrics.box.p.mean().item()
    recall = metrics.box.r.mean().item()
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    result = {
        "conf": conf,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50-95": round(map5095, 4)
    }

    results_list.append(result)

    print(
        f"conf={conf} | "
        f"P={result['precision']} | "
        f"R={result['recall']} | "
        f"F1={result['f1']} | "
        f"mAP50={result['mAP50']} | "
        f"mAP50-95={result['mAP50-95']}"
    )

print("\n===== Final Threshold Sweep Results =====\n")
for r in results_list:
    print(r)

# Best thresholds
best_f1 = max(results_list, key=lambda x: x["f1"])
best_precision = max(results_list, key=lambda x: x["precision"])
best_map = max(results_list, key=lambda x: x["mAP50-95"])

print("\n===== Best Threshold Summary =====")
print(f"Best F1 threshold        : conf={best_f1['conf']}  -> F1={best_f1['f1']}")
print(f"Best Precision threshold : conf={best_precision['conf']}  -> Precision={best_precision['precision']}")
print(f"Best mAP50-95 threshold  : conf={best_map['conf']}  -> mAP50-95={best_map['mAP50-95']}")

import csv

csv_path = r"D:\PLADICS\results\tables\threshold_sweep.csv"

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["conf", "precision", "recall", "f1", "mAP50", "mAP50_95"])

    for r in results_list:
        writer.writerow([
            r["conf"],
            r["precision"],
            r["recall"],
            r["f1"],
            r["mAP50"],
            r["mAP50-95"]
        ])

print(f"\nResults saved to {csv_path}")