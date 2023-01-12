import easyocr
import cv2
import numpy as np
from detect import detect

img = '1.jpg'

reader = easyocr.Reader(['en'], gpu=False)


x1, x2, y1, y2 = detect(img, 'best.pt', 640, 0.4, 0, 'cpu')
cv_img = cv2.imread(img)

plate_img = cv_img[y1:y2, x1:x2]
#cv2.imwrite(save_path, im0)


# image cleaning

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(plate_img)



results = reader.readtext(gray_image)
#xmin, xmax, ymin, ymax = results[0][0][0]
#cropped_img = gray_image[xmin:xmax, ymin:ymax]
print(results)

cv2.imshow('image', gray_image)
#cv2.imshow('image', cropped_img)
cv2.waitKey(0)


