import cv2
import numpy as np

img = cv2.imread("phuong.jpg")
img = np.float32(img) / 255.0

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# cv2.imwrite("gx.jpg", gx * 255)
# cv2.imwrite("gy.jpg", gy * 255)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imwrite("mag.jpg", mag * 255)
