import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/person_010.bmp", 0)
img_gaussian = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img_gaussian, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()