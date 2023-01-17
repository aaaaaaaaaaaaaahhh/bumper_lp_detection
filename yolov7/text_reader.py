import easyocr
import cv2
import numpy as np
from detect import detect

img = '1.jpg'

reader = easyocr.Reader(['en'], gpu=False)


x1, x2, y1, y2 = detect(img, 'best.pt', 640, 0.4, 0, 'cpu', view_img=False)
cv_img = cv2.imread(img)

plate_img = cv_img[y1:y2, x1:x2]
#cv2.imwrite(save_path, im0)


# image cleaning

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(plate_img)

img_blur = cv2.GaussianBlur(gray_image, (3,3), 0) 

# Canny Edge Detection
edges = cv2.Canny(image=gray_image, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image


thresh = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours = contours[0] if len(contours) == 2 else contours[1]
#big_contour = max(contours, key=cv2.contourArea)


result = np.zeros_like(edges)
cv2.drawContours(result, contours[5], -1, (255,255,255), cv2.FILLED)

cv2.imshow('result', result)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

results = reader.readtext(plate_img)
#xmin, xmax, ymin, ymax = results[0][0][0]
#cropped_img = gray_image[xmin:xmax, ymin:ymax]
print(results)

#cv2.imshow('image', gray_image)
#cv2.imshow('image', cropped_img)
#cv2.waitKey(0)


