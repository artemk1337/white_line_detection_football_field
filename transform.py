import cv2 as cv
import numpy as np


def add_text(image: np.array, im_pts=[[783, 501], [1021, 464], [1647, 715], [1909, 661]],
             zone=1, draw_lines=False):
    # image - photo of field
    # im_pts - 4 points of rectangle
    # zone - 1: gate rectangle
    # zone - 2: shtrafnaya rectangle
    # draw_lines - drawing field lines on image (if False - only words will be drawn)

    field = cv.imread('field.png', cv.IMREAD_COLOR)
    words = cv.imread('words.png', cv.IMREAD_COLOR)

    gate_pts = [[1063, 286], [1135, 286], [1063, 510], [1135, 510]]
    shtraf_pts = [[920, 150], [1135, 150], [920, 646], [1135, 646]]

    if (zone == 1):
        fld_pts = gate_pts
    if (zone == 2):
        fld_pts = shtraf_pts

    # for i in range(4):
    #     cv.line(image, (im_pts[i][0], im_pts[i][1]), (im_pts[i][0], im_pts[i][1]), (i * 64, i * 64, i * 64), thickness=10)
    #     cv.line(field, (fld_pts[i][0], fld_pts[i][1]), (fld_pts[i][0], fld_pts[i][1]), (i * 64, i * 64, i * 64),
    #             thickness=10)

    dst_pts = np.float32(im_pts).reshape(-1, 1, 2)
    src_pts = np.float32(fld_pts).reshape(-1, 1, 2)
    matrix = (cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)[0])
    words_t = cv.warpPerspective(src=words, M=matrix, dsize=image.shape[:2][::-1])
    field_t = cv.warpPerspective(src=field, M=matrix, dsize=image.shape[:2][::-1])
    if (not draw_lines):
        words_t = words_t - field_t

    # cv.imshow("1.png", cv.bitwise_or(image, words_t))

    return cv.bitwise_or(image, words_t)
