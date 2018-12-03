import cv2

img = cv2.imread("phuong.jpg")
print(img.shape)
img = cv2.resize(img, (64, 128))
print(img.shape)
hog = cv2.HOGDescriptor()
print(dir(hog))
h = hog.compute(img)
print(h.shape)
