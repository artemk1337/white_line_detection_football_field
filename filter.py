import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os


def plot_image(img, cmap=None):
    plt.axis("off")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap=cmap)
    plt.show()


class ImageHolder:
    def __init__(self):
        self.src_img = None
        self.filename = None
        self.dst_img = None

    def load_img(self, filename: str):
        self.src_img = cv.imread(filename)
        self.src_img = cv.resize(self.src_img, (1920, 1080))
        self.filename = filename
        self.dst_img = None

    def _filterMark_(self, img):
        # img = cv.resize(img, (1920, 1080))
        kernel = np.ones((15, 15), np.uint8)

        hsv = cv.cvtColor(cv.GaussianBlur(img, (5, 5), 5), cv.COLOR_BGR2HSV)
        mask_green = cv.inRange(hsv, (20, 20, 60), (70, 235, 200))

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

    def _a_b_x_y_list_(self, lines):
        a_b_x_y = []
        a_b = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if not (x2 - x1):
                break
            a = (y2 - y1) / (x2 - x1)
            if abs(a) < 5:
                b = y1 - a * x1
                a = np.arctan(a) * 180
                a_b += [[a, b]]
                a_b_x_y += [[a, b, line[0]]]
        return a_b_x_y, a_b

    def _create_clusters_(self, a_b):
        clustering = DBSCAN(eps=30, min_samples=1)
        clusters = clustering.fit_predict(a_b)
        clustersU = np.unique(clusters)
        return clusters, clustersU

    def _create_cluster_dict_(self, a_b_x_y, clusters):
        clusters_dict = {}
        for cluster in clusters: clusters_dict[cluster] = []
        for i, cluster in enumerate(clusters):
            a, b, [x1, y1, x2, y2] = a_b_x_y[i]
            clusters_dict[cluster] += [[a, b, [x1, y1, x2, y2]]]
        return clusters_dict

    def _find_best_points_(self, clusters_dict, clustersU, plot=True):
        total_points = []

        for cluster in clustersU:
            zeros = np.zeros((self.src_img.shape[0], self.src_img.shape[1])).astype(np.uint8)

            pair1, pair2 = [], []
            a_b_mean = []

            for line in clusters_dict[cluster]:
                a, b, [x1, y1, x2, y2] = line
                pair1 += [[x1, y1]]
                pair2 += [[x2, y2]]
                a_b_mean += [[a, b]]
                cv.line(zeros, (x1, y1), (x2, y2), 255, 2)

            #             print(a, b, x1, y1, x2, y2)
            #             plt.plot(a, b, ".")
            #         plt.show()

            #         plot_image(zeros)
            #         return -1

            a_b_mean = np.mean(a_b_mean, axis=0)
            # a_b_mean = a_b_mean[0]

            min_dist, best_points = 0, []
            for x1, y1 in pair1:
                for x2, y2 in pair2:
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist > min_dist:
                        min_dist = dist
                        best_points = [x1, y1, x2, y2]

            x1, y1, x2, y2 = best_points
            if plot:
                cv.line(zeros, (x1, y1), (x2, y2), 255, 2)
                plot_image(zeros)
            a, b = a_b_mean

            total_points += [[a, b, best_points]]
        return total_points

    def apply_filters(self, filename=None):
        if filename:
            self.load_img(filename)
        img = self.src_img
        img = self._filterMark_(img)

        morph = cv.morphologyEx(img, cv.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
        gray = morph[:, :, np.argmax(morph.mean(axis=(0, 1)))]

        kernel_size = 5
        gauss = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        edges = cv.Canny(gauss, 50, 150)

        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=180, maxLineGap=50)
        if not edges.mean():
            return None, None

        # zeros = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        # kernal_size = 15
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernal_size, kernal_size))
        # morph2 = cv.morphologyEx(zeros, cv.MORPH_CLOSE, kernel).astype(np.uint8)

        a_b_x_y, a_b = self._a_b_x_y_list_(lines)

        clusters, clustersU = self._create_clusters_(a_b)

        clusters_dict = self._create_cluster_dict_(a_b_x_y, clusters)

        total_points = self._find_best_points_(clusters_dict, clustersU, False)

        self.dst_img = self.src_img.copy()
        # self.dst_img = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        for a, b, [x1, y1, x2, y2] in total_points:
            cv.line(self.dst_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.save_image()
        return total_points, self.dst_img

    def save_image(self):
        if not os.path.exists("res_images"):
            os.mkdir("res_images")
        # print("res_images" + self.filename.split('/')[-1])
        cv.imwrite("res_images/" + self.filename.split('/')[-1], self.dst_img)
