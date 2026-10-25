import cv2 as cv
import numpy as np
from itertools import combinations
from math import lgamma, log, exp
from pathlib import Path

def match_probability(ni, nf, p1=0.6, p0=0.1, prior=1e-6):
    if nf == 0:
        return 0.0

    def log_binom(k, n, p):
        return (
            lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
            + k * log(p) + (n - k) * log(1 - p)
        )

    log_true = log_binom(ni, nf, p1) + log(prior)
    log_false = log_binom(ni, nf, p0) + log(1 - prior)
    x = log_false - log_true

    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1 / (1 + exp(x))

def find_homography(img1, img2, ratio=0.65):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    flann = cv.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    nf = len(good)

    if nf < 4:
        return None, nf, 0, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    ni = int(mask.sum()) if mask is not None else 0
    prob = match_probability(ni, nf)

    return H, nf, ni, prob

def stitch_to_middle(images):
    # 假设顺序为 A, B, C，以 B 为参考坐标系
    A, B, C = images
    H_AB, *_ = find_homography(A, B)  # A -> B
    H_CB, *_ = find_homography(C, B)  # C -> B

    transforms = [H_AB, np.eye(3), H_CB]

    corners_all = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
        corners_all.append(cv.perspectiveTransform(corners, H))

    corners_all = np.vstack(corners_all)
    xmin, ymin = np.floor(corners_all.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(corners_all.max(axis=0).ravel()).astype(int)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    out_w, out_h = xmax - xmin, ymax - ymin

    acc = np.zeros((out_h, out_w, 3), np.float32)
    cnt = np.zeros((out_h, out_w, 1), np.float32)

    for img, H in zip(images, transforms):
        warped = cv.warpPerspective(img, T @ H, (out_w, out_h))
        mask = (warped.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        acc += warped.astype(np.float32) * mask
        cnt += mask

    panorama = acc / np.maximum(cnt, 1)
    return panorama.astype(np.uint8)

if __name__ == "__main__":
    paths = ["1.jpg", "2.jpg", "3.jpg"]
    imgs = [cv.imread(p) for p in paths]

    for (i, img1), (j, img2) in combinations(enumerate(imgs), 2):
        H, nf, ni, p = find_homography(img1, img2)
        print(f"{paths[i]} - {paths[j]}: nf={nf}, ni={ni}, P={p:.6f}")

    pano = stitch_to_middle(imgs)
    cv.imwrite("panorama.jpg", pano)