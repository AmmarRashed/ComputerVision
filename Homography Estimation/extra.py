import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_score(Sxx, Syy, Sxy, alpha):
    """
    :param Sxx: sum of squares of the Ixx block (lambda 1 in the equation)
    :param Sxy: sum of squares of the Ixy block
    :param Syy: sum of squares of the Iyy block (lambda 2 in the equation)
    :param alpha: Harris detector free parameter in the equation (0.04 to 0.06)

    :return: Corner (keypoint) response
    """
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    return det - alpha * (trace ** 2)

def shi_tomasi(gray, blocksize=3, ksize=3, dilate=True):
    dst = cv2.cornerMinEigenVal(gray, blocksize, ksize)

    if dilate:
        return cv2.dilate(dst, None)
    return dst


def _manual_harris(gray, blocksize, alpha, dilate):

    h, w = gray.shape
    left_offset = blocksize//2
    right_offset = int(np.ceil(blocksize/2))

    dx, dy = np.gradient(gray)  # calculating x and y derivatives

    Ixx, Ixy, Iyy = dx ** 2, dx * dy, dy ** 2
    block_SS = lambda M, x, y: M[y-left_offset:y+right_offset, x-left_offset:x+right_offset].sum()

    dst = np.zeros((h, w))
    for y in range(left_offset, h-right_offset+1):
        for x in range(left_offset, w-right_offset+1):
            Sxx = block_SS(Ixx, x, y)
            Sxy = block_SS(Ixy, x, y)
            Syy = block_SS(Iyy, x, y)
            dst[(y, x)] = harris_score(Sxx, Syy, Sxy, alpha)

    if dilate:
        return cv2.dilate(dst, None)
    return dst


def _cv2_harris(gray, blocksize, ksize, alpha, dilate):
    dst = cv2.cornerHarris(gray, blocksize, ksize, alpha)

    if dilate:
        return cv2.dilate(dst, None)

    return dst


def harris(gray, blocksize=3, ksize=3, alpha=0.04, dilate=True, use_cv=True):
    """
    :param blocksize: window size, Default (3)
    :param ksize - Aperture parameter of Sobel derivative used, Default (3)
    :param alpha: Harris detector free parameter in the equation (0.04 to 0.06)
    :param dilate: result is dilated for marking the corners, not important
    :param use_cv: use opencv implementation or not

    :return: dst: array same shape as the gray image, with the keypoint response of each pixel
    """

    if use_cv:
        return _cv2_harris(gray, blocksize, ksize, alpha, dilate)
    return _manual_harris(gray, blocksize, alpha, dilate)


def annotate_keypoints(img, dst, threshold=0.05, show=False, negative=False):
    """
    :param dst: array same shape as the gray image, with the keypoint response of each pixel
    :param threshold: Corner (keypoint) response threshold as a ratio of the max response, Default (0.01)
    threshold = 0.01
    R > 0.01 * max(Rij for i in Y and j in X)
    :param show: draw the annotated image
    :param negative: annotate keypoints with red dots, or change the contrast between the image and the keypoints
        (i.e. keypoints will be white, but the rest of the image will be darker)

    :return: (iff show) image with keypoints annotated/highlighted
    """
    annotated = img.copy()
    r_threshold = dst.max() * threshold
    h, w = dst.shape

    if negative:
        annotated[dst <= r_threshold] = annotated[dst <= r_threshold] * 0.1
        annotated[dst > r_threshold] = [255, 255, 255]
    else:
        annotated[dst>r_threshold] = [255, 0, 0]

    if show:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(annotated)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    else:
        return annotated


def subpixel_accuracy(self, dst, threshold=0.01):
    """
    :param dst: array same shape as the gray image, with the keypoint response of each pixel
    :param threshold: Corner (keypoint) response threshold as a ratio of the max response, Default (0.01)
    threshold = 0.01
    R > 0.01 * max(Rij for i in Y and j in X)

    :return:
    """
    gray = self.get_gray()
    ret, dst = cv2.threshold(dst, threshold * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return centroids, corners
