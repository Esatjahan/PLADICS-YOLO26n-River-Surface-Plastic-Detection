from ultralytics import YOLO

model = YOLO(
    r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
)

for i, layer in enumerate(model.model.model):
    print(i, layer)