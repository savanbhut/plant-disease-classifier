import torch
from torchvision import models
import torch.nn as nn

num_classes = 38
model_check = models.mobilenet_v2(weights=None)
model_check.classifier[1] = nn.Linear(model_check.last_channel, num_classes)

state_dict = torch.load('plant_disease_model.pth', map_location='cpu')
model_check.load_state_dict(state_dict)
print("Loaded successfully")