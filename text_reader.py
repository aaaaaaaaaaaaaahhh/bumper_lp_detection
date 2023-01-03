import easyocr
import numpy as np
import torch

img = 'register-car-florida-image-1166.jpg'

reader = easyocr.Reader(['en'], gpu=False)

model = torch.hub.load('yolov7', 'custom', source='local', path='best.pt')
