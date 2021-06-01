import cv2 as cv
import numpy as np
from filter import ImageHolder
import multiprocessing




# multiprocessing.Process()


imh = ImageHolder("images/3.jpg")
_, im = imh.apply_filters()
cv.imshow("test", im)
cv.waitKey()

