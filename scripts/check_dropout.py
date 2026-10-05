from ultralytics import YOLO
import torch.nn as nn

model_path = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"

model = YOLO(model_path)
net = model.model

dropout_layers = []

for name, module in net.named_modules():
    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        dropout_layers.append(name)

print("Total dropout layers:", len(dropout_layers))

if dropout_layers:
    print("Dropout layers:")
    for layer in dropout_layers:
        print(layer)
else:
    print("No dropout layer found. Real MC Dropout is not available in this YOLO model.")