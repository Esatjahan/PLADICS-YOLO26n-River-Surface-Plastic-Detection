from ultralytics import YOLO

model_path = r"D:\PLADICS\experiments_precision\yolo26n_v2\weights\best.pt"
data_path = r"D:\PLADICS\merged_dataset_v2\plastic.yaml"

best_conf = 0.3

model = YOLO(model_path)

print("\n===== Final Test Evaluation at Best Threshold =====\n")

metrics = model.val(
    data=data_path,
    split="test",
    conf=best_conf,
    verbose=False
)

precision = metrics.box.p.mean().item()
recall = metrics.box.r.mean().item()
map50 = metrics.box.map50
map5095 = metrics.box.map

if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0

print(f"Best Threshold (from val): {best_conf}")
print(f"Test Precision   : {precision:.4f}")
print(f"Test Recall      : {recall:.4f}")
print(f"Test F1 Score    : {f1:.4f}")
print(f"Test mAP50       : {map50:.4f}")
print(f"Test mAP50-95    : {map5095:.4f}")