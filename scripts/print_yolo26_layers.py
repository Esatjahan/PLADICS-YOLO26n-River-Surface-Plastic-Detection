from ultralytics import YOLO

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"

model = YOLO(MODEL_PATH)
net = model.model

for i, layer in enumerate(net.model):
    print(i, layer.__class__.__name__, layer)