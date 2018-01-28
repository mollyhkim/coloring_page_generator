import numpy as np
import cv2 
from matplotlib import pyplot as plt

im = cv2.imread('Bear.jpg',True)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
#blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
contours = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

cv2.drawContours(thresh, contours, -1, (0, 0, 255), 5)

# display original image with contours
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", cv2.bitwise_not(thresh))
cv2.waitKey(0)