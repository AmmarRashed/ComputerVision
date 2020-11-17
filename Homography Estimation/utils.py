# Ammar Rasid S017288 MSc Computer Science (GSSE)

import os, ntpath
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from joblib import delayed, Parallel  # to parallelize RANSAC (only if the library is available)
    import multiprocessing

except ModuleNotFoundError:
    JOBLIB_EXISTS = False
else:
    JOBLIB_EXISTS = True


import imghdr  # for checking whether a file is a valid image file (by checking the file's magic number)

np.random.seed(42)

print("OpenCV version: "+cv2.__version__)
BOLD = '\033[1m'
END = '\033[0m'


def dst2kps(dst, max_kps):
    """converting dst to keypoints indices then to keypoints objects"""
    threshold = 0.1
    while True:
        kps_idx = list(map(lambda x: tuple([int(i) for i in x]), np.argwhere(dst > (dst.max() * threshold))))
        if max_kps<0 or len(kps_idx) <= max_kps: break
        threshold += 0.02
    return cv2.KeyPoint_convert(kps_idx)


def get_top_k(kp, k):
    """
    :param kp: lest of keypoints
    :param k: maximum number of keypoints
    :return: top k keypoints sorted by their response
    """
    return sorted(kp, key=lambda k:k.response, reverse=True)[:k]


def plot_prec_recall(scenes_precisions_recalls, detector, descriptor):
    for imgname in scenes_precisions_recalls:
        scene_name, filename = imgname.split("+")
        filename += "_{0}_{1}".format(detector, descriptor)
        precisions, recalls, thresholds = scenes_precisions_recalls[imgname]

        f, ax = plt.subplots(figsize=(12, 7))
        xticks = np.round(thresholds[:len(precisions)], 3)
        ax.plot(thresholds[:len(precisions)], precisions, '--o')
        ax.plot(thresholds[:len(precisions)], recalls, '--o')
        if len(xticks) < 16:
            ax.set_xticks(xticks)
        plt.suptitle(imgname)
        plt.title("{0}-{1}".format(detector, descriptor))
        ax.legend(["Precision", "Recall"])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        if not os.path.isdir("graphs"):
            os.makedirs("graphs")
        if not os.path.isdir(os.path.join("graphs", scene_name)):
            os.makedirs(os.path.join("graphs", scene_name))

        plt.savefig(os.path.join("graphs", scene_name, filename))
        plt.close(f)
        # plt.show()


def kps_union(kps):
    """
    :param kps: list of keypoint sets, each set corresponds to keypoints detected by a different detector
    :return: union of keypoints sets
    """
    kps_dict = dict()  # {point: kp object}
    for kp_set in kps:
        for kp in kp_set:
            kps_dict.setdefault(kp.pt, kp)

    return list(kps_dict.values())

class KeypointDetector:
    def __init__(self, inpath, scale=1.0):
        """
        :param inpath: path to the input image
        """
        self.inpath = inpath
        self.descriptors = {"sift": cv2.xfeatures2d.SIFT_create, "surf":cv2.xfeatures2d.SURF_create,
                            "brief":cv2.xfeatures2d.BriefDescriptorExtractor_create, "orb":cv2.ORB_create}
        self.img = cv2.imread(inpath)
        self.scale = scale
        self.gray = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)

    def get_gray(self):
        image = cv2.imread(self.inpath, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale)

    def good_features(self, use_harris, max_kps=-1):
        """
        :param use_harris: bool use harris or not
        :return: keypoints detected using Shi_Tomasi MinEigenVal or Harris detector if use_harris
        """
        if max_kps == -1:
            max_kps = 75*np.product(self.gray.shape)//100
        return cv2.KeyPoint_convert(
            cv2.goodFeaturesToTrack(self.gray, max_kps, 0.01, 10, useHarrisDetector=use_harris)
        )

    def draw_kp(self, kp):
        annotated = cv2.drawKeypoints(self.img, kp, np.zeros_like(self.img), color=(255, 0, 0))

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(annotated)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def fast(self, verbose=True, max_kps=1000, **kwargs):

        fast = cv2.FastFeatureDetector_create(**kwargs)

        gray = self.get_gray()
        kp = fast.detect(gray, None)
        kp = get_top_k(kp, max_kps)
        if verbose:
            print("Threshold: ", fast.getThreshold())
            print("nonmaxSuppression: ", fast.getNonmaxSuppression())
            print("neighborhood: ", fast.getType())
            print("Total Keypoints with{0} nonmaxSuppression: ".format("" if fast.getNonmaxSuppression() else "out"), len(kp))

        return kp

    def star(self, max_kps=1000, **kwargs):
        star = cv2.xfeatures2d.StarDetector_create(**kwargs)

        gray = self.get_gray()
        kp = star.detect(gray, None)
        kp = get_top_k(kp, max_kps)
        return kp

    def describe(self, descriptor_name, kp=None, **kwargs):
        descriptor = self.descriptors[descriptor_name](**kwargs)
        gray = self.get_gray()
        if kp is None:
            kp, desc = descriptor.detectAndCompute(gray, None)
        else:
            kp, desc = descriptor.compute(gray, kp, None)
        print("Keypoints count: {0:,}".format(len(kp)))
        return kp, desc


def filter_matches(matches, query_kps, train_kps, threshold):
    """
    :param l2norms: list of [(keypoint in img1, keypoint in img2, l2 norm of their descriptors)]
    :param threshold: matched descritpros are those whose L2 norm is smaller than threshold
    :return: list of query keypoints, list of train keypoints
    """
    filtered_query_kps, filtered_train_kps = list(), list()

    for m in matches:
        if m.distance < threshold:
            filtered_query_kps.append(query_kps[m.queryIdx])
            filtered_train_kps.append(train_kps[m.trainIdx])
    return np.float32(filtered_query_kps).reshape(-1, 1, 2), np.float32(filtered_train_kps).reshape(-1, 1, 2)


def match_features(img1_path, img2_path, detector="Harris", descriptor="SIFT", asarray=True, scale=1.0, **kwargs):
    """
    :param img1_path String path to input query image
    :param img2_path String path to input train image
    :param detector: keypoint detector name
    :param descriptor: list of [descriptor names]
    :param max_kps: max number of keypoints
    :param asarray: return keypoints as array or as Keypoint objects
    :return: list of [(keypoint in img1, keypoint in img2, l2 norm of their descriptors)]
    """
    print("\nQuery Image")
    query_kps, des1 = calculate_descs(img1_path, detector, descriptor, scale, **kwargs)  # query
    print("\nTrain Image")
    train_kps, des2 = calculate_descs(img2_path, detector, descriptor, scale, **kwargs)  # train
    print()

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    if asarray:
        query_kps = np.float32([k.pt for k in query_kps])
        train_kps = np.float32([k.pt for k in train_kps])
    return matches, query_kps, train_kps


def calculate_descs(img, detector, descriptor, scale, **kwargs):
    """
    :param img_path: path to the input image, or KeypointDetector object of the image
    :param detector: Keypoint detector "Harris", "Shi-Tomasi", "FAST", "Star"
    :param descriptor: descriptor "SIFT", "SURF", "BRIEF", "ORB"
    :param max_kps: max number of keypoints
    :param scale: scale of image
    :return keypoints objects, descriptors vectors
    """
    img_kpd = img if isinstance(img, KeypointDetector) else KeypointDetector(img, scale)

    if detector is None:
        kps, desc = img_kpd.describe(descriptor.lower())
    else:
        # keypoint/corner detection
        if detector.lower() == "fast":
            kps = img_kpd.fast(verbose=False, **kwargs)
        elif detector.lower() == "star":
            kps = img_kpd.star(**kwargs)
        else:
            kps = img_kpd.good_features(detector.lower() == "harris", **kwargs)

        print("Using {0}, {1:,} keypoints detected".format(detector, len(kps)))

        # calculating descriptors
        kps, desc = img_kpd.describe(descriptor.lower(), kps)

    # kps = [k.pt for k in kps]
    kps, desc = [list(t) for t in zip(*sorted(zip(kps, desc), key=lambda x: x[0].pt))]
    desc = np.array(desc)
    # kps = np.float32(kps).reshape(-1, 1, 2)

    return kps, desc


def evaluate_transformation(src_points, dst_points, max_distance=2):
    """
    :param src_points: source keypoints
    :param dst_points: destination keypoints
    :return: number of poitns that are transformed "near" the source image
    """
    total = 0
    for (x1, y1), (x2, y2) in zip(src_points.reshape(-1, 2), dst_points.reshape(-1, 2)):
        # Manhattan distance
        # if abs(x1-x2) + abs(y1-y2) < max_distance:
        #     total += 1
        if (x1-x2)**2 + (y1-y2)**2 < max_distance**2:
            total += 1
    return total


def calculate_precision_recall(ground, estimated, max_distance=2, verbose=False):
    """
    :param ground: output of transformation with ground truth H matrix
    :param dst: output of transformation with estimated H matrix
    :return: precision, recall
    """
    tp = 0.0
    fp = 0.0
    fn = 0.0
    ground = ground.reshape(-1, 2)
    estimated = estimated.reshape(-1, 2)
    closest = dict()  # {n: (m, distance)}
    for n, (x1, y1) in enumerate(estimated):
        for m, (x2, y2) in enumerate(ground):
            # distance = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
            distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            closest.setdefault(n, (m, distance))
            if distance < closest[n][1]:
                closest[n] = (m, distance)

    for n, (m, distance) in closest.items():
        if n == m:
            if distance < max_distance:
                tp += 1
            else:
                fn += 1
        elif distance < max_distance:
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    if verbose:
        print("Precision:", precision)
        print("Recall:", recall)
        print()
    return precision, recall


def estimate_homography_matrix(query_kps, train_kps, iterations=1e6, parallelize=True, ground=False):
    """
    :param matches: list of matched features
    :param query_kps: keypoints from image 1 query
    :param train_kps: keypoints from image 2 train
    :param iterations: number of iterations for RANSAC
    :param parallelize: parallelize RANSAC
    :return: estimated Homography matrix using RANSAC
    """
    if ground:
        h, mask = cv2.findHomography(query_kps, train_kps, cv2.RANSAC, 5.0)
        return h

    if JOBLIB_EXISTS and parallelize:
        cpu_count = multiprocessing.cpu_count()
        print("Parallelizing RANSAC over {0} processors".format(cpu_count))
        results = Parallel(-1)(delayed(_estimate_homography_matrix)(query_kps, train_kps, iterations//cpu_count) for _ in range(cpu_count))
        H, mask, max_similarity = max(results, key=lambda x: x[-1])
        print("Max similarity: ", max_similarity)
        return H, mask

    H, mask, max_similarity = _estimate_homography_matrix(query_kps, train_kps, iterations)
    print("Max similarity: ", max_similarity)
    return H, mask


def _estimate_homography_matrix(query_kps, train_kps, iterations):
    max_similarity = 0
    best_H = np.zeros((3, 3))
    best_mask = None
    for _ in range(int(min(iterations, len(query_kps)**3))):
        i, j, k = np.random.randint(0, len(query_kps), 3)
        if i == j and j == k: continue
        src_pts = query_kps[[i, j, k]]
        dst_pts = train_kps[[i, j, k]]

        H, mask = cv2.findHomography(src_pts, dst_pts)

        try:
            dst = cv2.perspectiveTransform(query_kps, H)

            similarity = evaluate_transformation(train_kps, dst)
            if similarity > max_similarity:
                max_similarity = similarity
                best_H = H
                best_mask = mask
        except Exception as e:
            # print(len(query_kps), len(train_kps))
            # print(e)
            continue
    return best_H, best_mask, max_similarity


def parseH(inpath):
    """
    :param inpath: path to the
    :return: H matrix
    """
    with open(inpath) as f:
        m = np.zeros((3, 3))
        for row, l in enumerate(f):
            for column, d in enumerate(l.strip().split()):
                m[row, column] = np.float32(d)
    return m


def get_min_max_dist(matches, min_scale, max_scale):
    min_ = np.inf
    max_ = -np.inf
    for m in matches:
        if m.distance > max_:
            max_ = m.distance
        if m.distance < min_:
            min_ = m.distance
    return min_*min_scale, max_*max_scale


class HomographyEstimation:
    def __init__(self, indir):
        """
        :param indir: input images directory
        """
        self.indir = indir
        # [images pathes], [H files path]
        self.img_pathes, self.Hs = self.parse_data()

    def parse_data(self):
        imgs = list()
        Hs = list()
        for path in os.listdir(self.indir):
            filepath = os.path.join(self.indir, path)
            try:
                if imghdr.what(filepath):  # valid image file
                    imgs.append(filepath)
                else:
                    Hs.append(filepath)
            except Exception as e:
                if isinstance(e, IsADirectoryError):
                    continue
                print(e)
        return imgs, Hs

    def pipeline(self, detector, descriptor, thresholds_size=10, RANSAC_iterations=1e6, min_scale=1.0, max_scale=1.0,
                 verbose=False, scale=1.0, **kwargs):
        """
        :param detector: detector name (e.g. "Harris", "FAST", "Shi-tomasi", "STAR"
        :param descriptor: String descriptor name
        :param thresholds_size: number of thresholds to test
        :param RANSAC_iterations: maximum number of iterations for RANSAC
        :param min_scale: scale for the min matching distance to use as min threshold (>=1.0)
        :param max_scale: scale for the max matching distance to use as max threshold (<=1.0)
        :param verbose: print model statistics every iteration
        :param scale: scale image (change resolution)
        :param kwargs: kwargs for detector (max_kps int maximum number of keypoints by harris/shi-tomasi)
        :return: precision recall dictionary
            {scene image name: ([list of precisions per iteration, .. recall, .. thresholds])
        """
        query_images = sorted(self.img_pathes)
        query_img = query_images[0]  # query image (no transformation)
        train_images = query_images[1:]  # train images  (with transformation)

        h_pathes = sorted(self.Hs)

        scenes_precisions_recalls = dict()
        root_name = ntpath.basename(self.indir)

        for train_img, h_path in zip(train_images, h_pathes):
            name = root_name + ".".join(ntpath.basename(train_img).split('.')[:-1])
            print(name)
            start = time.time()
            precisions = list()
            recalls = list()
            h = parseH(h_path)

            matches, query_kps, train_kps = match_features(query_img, train_img, detector, descriptor, scale=scale, **kwargs)

            matched_query_kps, matched_train_kps = filter_matches(matches, query_kps, train_kps, float("inf"))
            X1_2 = cv2.perspectiveTransform(matched_query_kps, m=h)  # ground dst

            min_dist, max_dist = get_min_max_dist(matches, min_scale=min_scale, max_scale=max_scale)

            thresholds = np.linspace(min_dist, max_dist, thresholds_size)
            for threshold in sorted(thresholds, reverse=True):
                filtered_query_kps, filtered_train_kps = filter_matches(matches, query_kps, train_kps, threshold)
                print("Threshold: {0}\t{1:,} Matches".format(round(threshold,2), len(filtered_query_kps)))
                estimated_h, mask = estimate_homography_matrix(filtered_query_kps, filtered_train_kps, RANSAC_iterations)
                if estimated_h is None:
                    print("Not enough matches")
                    break
                try:
                    estimated_X1_2 = cv2.perspectiveTransform(matched_query_kps, m=estimated_h)  # estimate dst
                except Exception as e:
                    print(e)
                    continue

                precision, recall = calculate_precision_recall(X1_2, estimated_X1_2, verbose=verbose)
                precisions.append(precision)
                recalls.append(recall)
            scenes_precisions_recalls[root_name+"+"+name] = (precisions, recalls, thresholds)
            print("Took: ", time.time()-start)
            print()

        plot_prec_recall(scenes_precisions_recalls, detector, descriptor)
        return scenes_precisions_recalls


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", help="Images paths", required=True)
    parser.add_argument("-dt", help="Detectors", nargs="+", required=True)
    parser.add_argument("-ds", help="descriptors", nargs="+", required=True)
    parser.add_argument("-s", help="descriptors", default=1.0, type=float)
    parser.add_argument("-v", help="Verbose", default=1, type=int)
    parser.add_argument("-i", help="RANSAC iterations", default=multiprocessing.cpu_count()*1e3, type=float)
    parser.add_argument("-t", help="Thresholds size", default=15, type=int)
    parser.add_argument("-m", help="Max number of Keypoints", default=100, type=int)

    args = parser.parse_args()

    he = HomographyEstimation(args.p)
    for descriptor in list(args.ds):
        for detector in list(args.dt):
            he.pipeline(detector,
                        descriptor,
                        RANSAC_iterations=args.i,
                        thresholds_size=args.t,
                        verbose=args.v,
                        scale=args.s,
                        max_kps=args.m)

# use like:
# python3 utils.py -p data/boat -dt Shi-tomasi FAST -ds SIFT SURF -s 0.5 -v 1 -i 8000 -t 20 -m 100