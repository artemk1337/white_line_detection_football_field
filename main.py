from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Test", gray)
# cv2.waitKey()


(thresh, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("Test", gray)
# cv2.waitKey()


kernel_size = 15
gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
# cv2.imshow("Test", gray)
# cv2.waitKey()


edges = cv2.Canny(gray, 50, 150)
# cv2.imshow("Test", edges)
# cv2.waitKey()


lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=75, maxLineGap=5)
print(lines.shape)

for line in lines:
    x1, x2, y1, y2 = line[0]
    cv2.line(img, (x1, x2), (y1, y2), (0, 0, 255), 2)

cv2.imshow("Test", img)
cv2.waitKey()
cv2.imwrite("RES.png", img)


# lines = cv2.HoughLines(edges, 1.5, np.pi/180, 50)
# for rho, theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# cv2.imshow("Test", img)
# cv2.waitKey()

# lines = probabilistic_hough_line(edges, threshold=10, line_length=100,
#                                  line_gap=10)
# plt.imshow(edges, cmap='gray')
# for line in lines:
#     p0, p1 = line
#     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
# plt.show()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# for i in range(3):
#     cv2.imshow("Test", img[:, :, i])
#     cv2.waitKey()


