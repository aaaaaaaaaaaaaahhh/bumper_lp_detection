import easyocr
import numpy as np
import torch

img = '1.jpg'

reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(img)


model = torch.hub.load('yolov7', 'custom', source='local', path='best.pt')

results = model(img)

labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

print(results)
