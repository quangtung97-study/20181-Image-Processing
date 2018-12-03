import cv2
import sys


def overlap(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    if x1 < x2 or x1 + w1 > x2 + w2:
        return False
    if y1 < y2 or y1 + h1 > y2 + h2:
        return False
    return True


def filter_founds(founds, weights):
    founds_result = []
    weights_result = []
    n = len(founds)
    for i in range(0, n):
        found = founds[i]
        weight = weights[i]
        filtered = False
        for j in range(0, n):
            if i == j:
                continue
            if overlap(found, founds[j]):
                filtered = True
                break
        if not filtered:
            founds_result.append(found)
            weights_result.append(weight)
    return (founds_result, weights_result)


img = cv2.imread(sys.argv[1])
hog = cv2.HOGDescriptor()

params = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(params)

founds, weights = hog.detectMultiScale(
    img,
    # hitThreshold=0,
    winStride=(8, 8),
    padding=(32, 32),
    scale=1.05,
    finalThreshold=2
)

print(founds)
print(weights)

founds, weights = filter_founds(founds, weights)

print(founds)
print(weights)

for i in range(0, len(founds)):
    weight = weights[i]
    if weight < 0.5:
        continue

    found = founds[i]
    x, y, w, h = found
    cv2.rectangle(
        img, (x, y),
        (x + w, y + h),
        (0, 255, 0), 3)

cv2.imwrite("draw.jpg", img)

cv2.imshow(sys.argv[1], img)
cv2.waitKey(0)
cv2.destroyAllWindows()
