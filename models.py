"""Model loading utilities."""
from ultralytics import YOLO
import torch

# Load YOLO once
yolo_model = YOLO('yolo11s.pt')

# Load MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()

device = torch.device("cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
