import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_JSON = r"D:\PLADICS\datasets_clean\merged_clean_dataset\annotations\instances_val.json"
PRED_JSON = r"D:\nanodet\workspace\pladics_nanodet_640_fair\nanodet_fresh_test_results.json"
FIXED_JSON = r"D:\nanodet\workspace\pladics_nanodet_640_fair\nanodet_fixed_val_results.json"
MODEL_PATH = r"D:\nanodet\workspace\pladics_nanodet_640_fair\model_best\nanodet_model_best.pth"

with open(GT_JSON, "r") as f:
    gt = json.load(f)

valid_ids = set(img["id"] for img in gt["images"])

with open(PRED_JSON, "r") as f:
    preds = json.load(f)

fixed = [p for p in preds if p["image_id"] in valid_ids]

with open(FIXED_JSON, "w") as f:
    json.dump(fixed, f)

print("Original predictions:", len(preds))
print("Filtered predictions:", len(fixed))
print("Removed predictions:", len(preds) - len(fixed))

coco_gt = COCO(GT_JSON)
coco_dt = coco_gt.loadRes(FIXED_JSON)

evaluator = COCOeval(coco_gt, coco_dt, "bbox")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()

precision_array = evaluator.eval["precision"]
recall_array = evaluator.eval["recall"]

p = precision_array[0, :, 0, 0, 2]
p = p[p > -1]
r = recall_array[0, 0, 0, 2]

precision_50 = float(np.mean(p)) if len(p) > 0 else 0.0
recall_50 = float(r)

size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n========== NanoDet Final COCOeval Metrics ==========")
print(f"Precision@IoU0.50: {precision_50:.4f}")
print(f"Recall@IoU0.50: {recall_50:.4f}")
print(f"mAP@0.5: {evaluator.stats[1]:.4f}")
print(f"mAP@0.5:0.95: {evaluator.stats[0]:.4f}")
print(f"Model Size: {size_mb:.2f} MB")
print("===================================================")