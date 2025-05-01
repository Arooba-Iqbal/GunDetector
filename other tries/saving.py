import torch

# Load model from original .pt (AutoShape or torch.save model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# ✅ Save in full YOLOv5 checkpoint format
model.model.save('best_yolov5.pt')  # `model.model` is the raw nn.Module
