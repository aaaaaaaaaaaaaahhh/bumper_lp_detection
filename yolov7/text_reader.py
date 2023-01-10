import easyocr
import numpy as np
from detect import detect

img = '7.jpg'

reader = easyocr.Reader(['en'], gpu=False)


detect(img, 'best.pt', 640, 0.4, 0, 'cpu')



