import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/person_010.bmp", 0)
img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.show()