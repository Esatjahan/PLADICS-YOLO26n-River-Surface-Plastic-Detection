from ultralytics import YOLO

model = YOLO("D:/PLADICS/models_custom/yolo26n_lightweight.yaml")

model.train(
    data="D:/PLADICS/merged_dataset_v2/plastic.yaml",
    imgsz=640,
    epochs=40,
    batch=8,
    device="cpu",
    workers=0,
    project="D:/PLADICS/experiments_precision",
    name="yolo26n_lightweight_clean"
)