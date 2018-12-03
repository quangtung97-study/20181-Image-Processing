import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/person_010.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(img,100,200)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=3)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=3)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewittx+img_prewitty

# roberts
kernelx = np.array([[0,1],[-1,0]])
kernely = np.array([[-1,0,],[0,1]])
img_robertx = cv2.filter2D(img_gaussian, -1, kernelx)
img_roberty = cv2.filter2D(img_gaussian, -1, kernely)
img_robert = img_robertx+img_roberty

"""
cv2.imshow("Original Image", img)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""


plt.subplot(151), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(152), plt.imshow(img_canny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])


plt.subplot(153), plt.imshow(img_sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(154), plt.imshow(img_prewitt, cmap='gray')
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])

plt.subplot(155), plt.imshow(img_robert, cmap='gray')
plt.title('Robert'), plt.xticks([]), plt.yticks([])

plt.show()