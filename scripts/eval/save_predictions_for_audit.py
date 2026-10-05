from ultralytics import YOLO
import os

model_path = r"D:\PLADICS\experiments_precision\yolo26n_v2\weights\best.pt"

val_images = r"D:\PLADICS\merged_dataset_v2\images\val"
test_images = r"D:\PLADICS\merged_dataset_v2\images\test"

val_save_dir = r"D:\PLADICS\analysis\fp_fn_audit\predictions_val"
test_save_dir = r"D:\PLADICS\analysis\fp_fn_audit\predictions_test"

model = YOLO(model_path)

os.makedirs(val_save_dir, exist_ok=True)
os.makedirs(test_save_dir, exist_ok=True)

print("Saving VAL predictions...")
model.predict(
    source=val_images,
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=True,
    project=val_save_dir,
    name="yolo26n_v2_val_preds",
    exist_ok=True
)

print("Saving TEST predictions...")
model.predict(
    source=test_images,
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=True,
    project=test_save_dir,
    name="yolo26n_v2_test_preds",
    exist_ok=True
)

print("Done.")