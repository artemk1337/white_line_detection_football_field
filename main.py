import cv2 as cv
import numpy as np


def filter(img):
    img = cv.resize(img, (1920, 1080))
    kernel = np.ones((15, 15), np.uint8)

    hsv = cv.cvtColor(cv.GaussianBlur(img, (5, 5), 5), cv.COLOR_BGR2HSV)
    mask_green = cv.inRange(hsv, (20, 20, 90), (70, 235, 200))

    mask_green = cv.morphologyEx(mask_green, cv.MORPH_OPEN, kernel)

    components = cv.connectedComponentsWithStats(mask_green, 4, cv.CV_32S)
    max_area_index = (np.argmax(components[2], axis=0))[4]
    for i in range(len(components[1])):
        for j in range(len(components[1][i])):
            if (components[1][i][j] != max_area_index):
                mask_green[i][j] = 0
    mask_green = cv.morphologyEx(mask_green, cv.MORPH_CLOSE, kernel)
    img = cv.bitwise_and(img, img, mask=mask_green)
    return img


original = cv.imread('2.jpg')

img = filter(original)
cv.imshow('Test1', img)
cv.waitKey()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Test", gray)
# cv.waitKey()

(thresh, gray) = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# cv.imshow("Test", gray)
# cv.waitKey()

# kernel_size = 15
# gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)
# cv.imshow("Test", gray)
# cv.waitKey()

edges = cv.Canny(gray, 50, 150)
cv.imshow("Test", edges)
cv.waitKey()

lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=75, maxLineGap=20)
print(lines.shape)

img = np.zeros((img.shape[0], img.shape[1]))
for line in lines:
    x1, x2, y1, y2 = line[0]
    cv.line(img, (x1, x2), (y1, y2), 255, 15)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel).astype(np.uint8)
cv.imshow("Test", img)
cv.waitKey()






############


#################################   SIFT   ####################################
# sift = cv.SIFT_create()
#
# img_kp, img_ds = sift.detectAndCompute(grayscale, None)
# field_kp, field_ds = sift.detectAndCompute(field, None)
#
# cv.drawKeypoints(original, img_kp, original, color=(255, 0, 0))
#
# cv.imshow('qwe', original)
# cv.waitKey()
#
# matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
#
# local_matches = matcher.knnMatch(field_ds, img_ds, k=2)
# good = []
# for m, mm in local_matches:
#     if m.distance < 0.75 * mm.distance:
#         good.append(m)
# matches = good
#
# src_pts = np.float32([img_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([field_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# matrix = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
# dst = cv.warpPerspective(src=original, M=matrix[0], dsize=(1200, 800))[0]
#
# cv.imshow('dst.jpg', dst)
# cv.waitKey()
