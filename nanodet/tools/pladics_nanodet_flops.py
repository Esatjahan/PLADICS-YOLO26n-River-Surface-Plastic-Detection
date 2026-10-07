import torch
from thop import profile

from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model


CONFIG_PATH = r"D:\nanodet\config\pladics_nanodet_official.yml"
CKPT_PATH = r"D:\nanodet\workspace\pladics_nanodet_640_fair\model_best\model_best.ckpt"

load_config(cfg, CONFIG_PATH)

model = build_model(cfg.model)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["state_dict"]

clean_state = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        clean_state[k.replace("model.", "", 1)] = v
    else:
        clean_state[k] = v

missing, unexpected = model.load_state_dict(clean_state, strict=False)

model.eval()

dummy = torch.randn(1, 3, 640, 640)

macs, params = profile(model, inputs=(dummy,), verbose=False)

# THOP gives MACs. Many papers report FLOPs ≈ 2 * MACs.
gmacs = macs / 1e9
gflops = (2 * macs) / 1e9
params_m = params / 1e6

print(f"Params(M) = {params_m:.2f}")
print(f"GMACs = {gmacs:.2f}")
print(f"GFLOPs = {gflops:.2f}")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))