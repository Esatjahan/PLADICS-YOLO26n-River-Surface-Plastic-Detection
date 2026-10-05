from ultralytics import YOLO
from YOLOv8_Explainer.core import yolov8_heatmap

model = YOLO(
    r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
)

yolov8_heatmap(
    weight=model,
    source=r"D:\PLADICS\xai_yolo26\input",
    method="GradCAM",
    layer=[22],
    conf_threshold=0.25,
    ratio=0.02,
    show_box=True,
    renormalize=True,
    save_path=r"D:\PLADICS\xai_yolo26\gradcam_output"
)