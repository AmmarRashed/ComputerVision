# Ammar Rasid S017288 Department of Computer Science, GSSE

from __future__ import print_function, absolute_import, division

import cv2
import numpy as np
import os
import argparse
import time
import pickle

PARALLELIZE = True
try:
    from joblib import Parallel, delayed
except ImportError:
    PARALLELIZE = False
    pass

LK_DEFAULT = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

HOOF_PARAMS_DEFAULT = {"outpath": "", "bins":4, "window":3, "lk_params":LK_DEFAULT}

def img2xy(gray, window=5):
    h, w = gray.shape
    return np.array([(x, y) for y in range(int(window/2), h, window) for x in range(int(window/2), w, window)]
                    , dtype=np.float32).reshape(-1, 1, 2)


def theta2b(theta, B):
    # normalize theta to   0 <= theta < 2 pi
    if theta < 0:
        theta += 2*np.pi * (int(abs(theta / np.pi/2))+1)
    elif theta > 2 * np.pi:
        theta = theta - (int(theta/np.pi/2) * np.pi * 2)

    # calculating the bin according to each of the 4 quarters
    if 0 <= theta < np.pi/2:
        return int(B/2. + B*theta/np.pi)
    if np.pi/2 <= theta < np.pi:
        theta = np.pi - theta
        return int(B / 2/ + B * theta / np.pi)

    if np.pi <= theta < 3 * np.pi / 2:
        theta = theta - np.pi
        return int(B / 2. - B * theta / np.pi)
    if 3 * np.pi / 2 <= theta <= 2*np.pi:
        theta = 2 * np.pi - theta
        return int(B / 2. - B * theta / np.pi)


def calculate_hoof(of_vectors, bins):
    hoof = np.zeros(bins)
    for x, y in of_vectors:
        theta = 0 if x == 0 else np.arctan(y/x)
        contribution = np.sqrt(x**2 + y**2)
        b = theta2b(theta, bins)
        hoof[b] += contribution
    return hoof


def calculate_temporal_mean_hoof(inpath, outpath, bins, window, lk_params):
    cap = cv2.VideoCapture(inpath)
    success, fr = cap.read()
    prev_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    prev_points = img2xy(prev_gray, window)
    hoof = np.zeros(bins)

    h, w = prev_gray.shape
    # Define the codec and create VideoWriter object
    if outpath:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(outpath, fourcc, 20.0, (w, h))
    i = 1
    while success:
        success, fr = cap.read()
        mask = np.zeros_like(fr)
        if not success: break
        new_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        new_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, new_gray, prev_points, None, **lk_params)
        of_v = list()  # optical flow vectors
        for prev, new in zip(prev_points, new_points):
            x1, y1 = prev.ravel()
            x2, y2 = new.ravel()
            new_v = np.float32([x2 - x1, y2 - y1])
            of_v.append(new_v)
            if outpath:
                mask = cv2.arrowedLine(mask, (x1, y1), (x2, y2), (0, 0, 255))
        of_v = np.array(of_v, dtype=np.float32)
        i += 1
        hoof += calculate_hoof(of_v, bins)
        if outpath:
            img = cv2.add(fr, mask)
            video.write(img)
    hoof /= i  # i is the number of frames
    if outpath:
        video.release()
    cap.release()
    return hoof


def train_knn(X_train, y_train):
    knn = cv2.ml.KNearest_create()
    knn.train(X_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32).reshape(-1, 1))
    return knn


def evaluate_knn(knn, X_test, y_test, k):
    """
    :param knn: knn model
    :param X_test:
    :param y_test:
    :param k: k value in KNN (number of neighbors)
    :return: predictions, accuracy
    """
    ret, predictions, neighbours, dist = knn.findNearest(X_test.astype(np.float32), k)
    accuracy = 0.
    with open("out.txt", 'w') as f:
        f.write("Prediction\tActual\n")
        for pred, actual in zip(predictions, y_test):
            accuracy += int(pred == actual)
            f.write("{0}\t{1}\n".format(pred, actual))
    accuracy /= len(predictions)
    return predictions, accuracy


def train_test_split(X, y, k):
    test_indices = list()
    for label in set(y):
        for i in np.random.choice(np.where(y==label)[0], size=k, replace=False):
            test_indices.append(i)

    X_test = X[test_indices]
    y_test = y[test_indices]

    train_indices = list(set(range(len(y))) - set(test_indices))

    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, y_train, X_test, y_test


class PCA:
    model = None

    def fit(self, X, energy=0.9):
        cov = np.cov(X.T)
        w, v = np.linalg.eig(cov)
        w /= sum(w)
        i = 0
        for e in w:
            i += 1
            energy -= e
            if energy <= 0:
                break
        self.model = v[:, :i]

    def transform(self, X):
        return X.dot(self.model)

def euc_distance(a, b):
    return np.linalg.norm(a-b)

def k_nearest_neighbour(X, y, newcomer, k):
    distances = [(0, i) for i in y]  # [(distance, label) for each feature vector in X]
    for i, v in enumerate(X):
        distances[i] = (euc_distance(newcomer, v), y[i])
    distances = sorted(distances, key=lambda v:v[0])
    return [distances[i][1] for i in range(k)]

def hoof_pipeline(root_path, hoof_params=HOOF_PARAMS_DEFAULT, test_size_per_class=1):
    """
    :param root_path: the folder containing all the scenes
    :param hoof_params: dictionary of parameters for hoof calculation method (calculate_temporal_mean_hoof)
    :param test_size_per_class: number of test samples per class
    :return: X_train, y_train, X_test, y_test
    """
    paths = list()
    labels = list()
    label2id = dict()
    id2label = dict()
    if not os.path.isdir("output"):
        os.makedirs("output")
    for scene in os.listdir(root_path):
        for sample in os.listdir(os.path.join(root_path, scene)):
            inpath = os.path.join(root_path, scene, sample)
            out_scene_path = os.path.join("output", scene)
            if not os.path.isdir(out_scene_path):
                os.makedirs(out_scene_path)
            outpath = os.path.join("output", scene, sample)
            paths.append((inpath, outpath))
            label2id.setdefault(scene, len(label2id))
            labels.append(label2id[scene])

            id2label.setdefault(label2id[scene], scene)
    y = np.array(labels, dtype=np.float32).reshape(-1)

    if PARALLELIZE:
        hoofs = Parallel(n_jobs=-1)(delayed(calculate_temporal_mean_hoof)(inpath, **hoof_params)
                                    for (inpath, outpath) in paths)
    else:
        hoofs = [calculate_temporal_mean_hoof(inpath, **hoof_params) for (inpath, outpath) in paths]
    hoofs = np.array(hoofs, dtype=np.float32)

    X_train, y_train, X_test, y_test = train_test_split(hoofs, y, test_size_per_class)

    return X_train, y_train, X_test, y_test


def grid_search(bins, ks, hoof_params, energy, test_size_per_class):
    """
    :param bins: A list of B (number of bins) to be used in HOOF
    :param ks: A list of K (number of neighbours) to be used in KNN algorithm
    :param hoof_params: A dictionary of HOOF calculation parameters
    :param energy: minimum energy in PCA
    :param test_size_per_class: number of test samples per class
    :return: a list of the accuracies and another of the runtimes of each combination of B and K
    """
    n = len(bins) * len(ks)
    runtimes = np.zeros(n)
    accuracies = np.zeros(n)
    best_model = (bins[0], ks[0])
    max_accuracy = 0
    i = 0
    for b in bins:
        for k in ks:
            print("Trying B: {0} \t K: {1}".format(b, k))
            start = time.time()
            hoof_params["bins"] = b
            X_train, y_train, X_test, y_test = hoof_pipeline("data",
                                                             hoof_params,
                                                             test_size_per_class=test_size_per_class)

            pca = PCA()
            pca.fit(X_train, energy=energy)

            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

            knn = train_knn(X_train, y_train)
            predictions, accuracy = evaluate_knn(knn, X_test, y_test, k)
            runtimes[i] = time.time() - start
            accuracies[i] = accuracy

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_model = (b, k)
            print("Took: {0}\nAccuracy: {1}".format(runtimes[i], accuracies[i]))
            i += 1
    print("Best Model: B:{0}\tK: {1}".format(*best_model))
    return accuracies, runtimes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-inpath", help="The path to the folder containing the entire data, "
                                        "each class in a separate folder", required=True, type=str)
    parser.add_argument("-Bs", help="A list of the number of bins (B) to use in HOOF", required=True, type=int,
                        nargs='+')
    parser.add_argument("-g", help="The step size (distance between each two subsequent pixels) in the optical flow grid", default=3,
                        type=int, required=False)
    parser.add_argument('-w', help="The window size to be used in Lucas-Kanade", default=15, type=int, required=False)

    parser.add_argument('-Ks', help="A list of K values to be used in KNN classification algorithm", default=3,
                        nargs='+', type=int, required=False)
    parser.add_argument('-e', help="Minimum energy to preserve in PCA (btw 0-1)", default=0.9, type=float,
                        required=False)
    parser.add_argument('-t', help="Number of test samples per class", type=int, required=False, default=1)

    args = parser.parse_args()


    lk_params = LK_DEFAULT.copy()
    lk_params["winSize"] = (args.w, args.w)
    hoof_params = {"bins": args.Bs, "window": args.g, "lk_params": lk_params, "outpath":""}

    accuracies, runtimes = grid_search(args.Bs, args.Ks, hoof_params, args.e, args.t)

    with open("performance.pkl", 'wb') as f:
        pickle.dump({"accuracies":accuracies, "runtimes":runtimes, "Bs":args.Bs, "Ks":args.Ks}, f, protocol=2)

# USAGE:
# python3 src.py -inpath data -Bs 2 3 4 5 -g 5 -w 15 -Ks 2 3 4 5 -e 0.9 -t 1
