import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('B2.jpg', 0)
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('B2.jpg', thresh)
cv2.waitKey(0)
