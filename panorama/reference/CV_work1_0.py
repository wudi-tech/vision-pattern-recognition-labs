import cv2 as cv
import numpy as np
from itertools import combinations
from math import lgamma, log, exp
from pathlib import Path


def largest_rectangle(binary):
    rows, cols = binary.shape
    heights = [0] * cols
    best_area = 0
    best_rect = (0, rows, 0, cols)

    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if binary[r, c] else 0

        stack = []
        for c in range(cols + 1):
            current_height = heights[c] if c < cols else 0
            start = c

            while stack and stack[-1][1] > current_height:
                idx, height = stack.pop()
                area = height * (c - idx)

                if area > best_area:
                    best_area = area
                    best_rect = (r - height + 1, r + 1, idx, c)

                start = idx

            stack.append((start, current_height))

    return best_rect

def crop_black_border(img, valid_mask, max_crop_side=900):
    binary = (valid_mask > 0).astype(np.uint8)
    h, w = binary.shape
    scale = min(1.0, max_crop_side / max(h, w))

    if scale < 1.0:
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small = cv.resize(binary, (small_w, small_h), interpolation=cv.INTER_NEAREST)
    else:
        small = binary

    top, bottom, left, right = largest_rectangle(small)

    if scale < 1.0:
        top = int(top / scale)
        bottom = int(np.ceil(bottom / scale))
        left = int(left / scale)
        right = int(np.ceil(right / scale))

    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))
    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))

    while top < bottom - 1 and binary[top, left:right].mean() < 0.99:
        top += 1
    while bottom > top + 1 and binary[bottom - 1, left:right].mean() < 0.99:
        bottom -= 1
    while left < right - 1 and binary[top:bottom, left].mean() < 0.99:
        left += 1
    while right > left + 1 and binary[top:bottom, right - 1].mean() < 0.99:
        right -= 1

    return img[top:bottom, left:right]

def probability(ni, nf, p1=0.6, p0=0.1, prior=1e-6):
    if nf == 0:
        return 0.0
    
    def log_binom(k, n, p):
        return(
            lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k*log(p)+(n-k)*log(1-p)
        )
    
    log_true = log_binom(ni, nf, p1) + log(prior)
    log_false = log_binom(ni, nf, p0) + log(1- prior)
    x = log_false - log_true

    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1/(1+exp(x))

def transform(img1, img2, ratio=0.75):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, 0, 0, 0.0

    flann = cv.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    nf = len(good)

    if nf < 4:
        return None, nf, 0, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)

    if H is None or mask is None:
        return None, nf, 0, 0.0

    ni = int(mask.sum())
    prob = probability(ni, nf)

    return H, nf, ni, prob

def image_stitching(images):
    A, B, C = images
    H_AB, *_ = transform(A, B)
    H_CB, *_ = transform(C, B)

    if H_AB is None or H_CB is None:
        raise ValueError("Not enough overlap to estimate homography.")

    transforms = [H_AB, np.eye(3), H_CB]

    corners_all = []

    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
        corners_all.append(cv.perspectiveTransform(corners,H))

    corners_all = np.vstack(corners_all)
    xmin, ymin =np.floor(corners_all.min(axis=0).ravel()).astype(int)
    xmax, ymax =np.ceil(corners_all.max(axis=0).ravel()).astype(int)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    out_w, out_h = xmax - xmin, ymax - ymin
    acc = np.zeros((out_h, out_w, 3), np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), np.float32)

    for img, H in zip(images, transforms):
        warped = cv.warpPerspective(img, T @ H, (out_w, out_h))
        src_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv.warpPerspective(src_mask, T @ H, (out_w, out_h), flags=cv.INTER_NEAREST)
        mask = (warped_mask > 0).astype(np.uint8)

        weight = cv.distanceTransform(mask, cv.DIST_L2, 3)
        if weight.max() > 0:
            weight = weight / weight.max()

        weight = weight[..., None].astype(np.float32)
        acc += warped.astype(np.float32) * weight
        weight_sum += weight
        
    panorama = acc / np.maximum(weight_sum, 1e-6)
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    valid_mask = (weight_sum[:, :, 0] > 1e-6).astype(np.uint8)
    return crop_black_border(panorama, valid_mask), panorama
    
if __name__ == "__main__":
    paths = ["3_1.jpg", "3_2.jpg", "3_3.jpg"]
    imgs = [cv.imread(p) for p in paths]

    for p, img in zip(paths, imgs):
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")

    #imgs = [resize_image(img) for img in imgs]

    for (i,img1), (j, img2) in combinations(enumerate(imgs), 2):
        H, nf, ni,p = transform(img1,img2)
        print(f"{paths[i]} - {paths[j]}: nf={nf}, ni={ni}, P={p:.6f}")

    pano1, pano2 = image_stitching(imgs)

    if not cv.imwrite("result3_0.jpg", pano1):
        raise RuntimeError("Failed to save result1.jpg")
    
    if not cv.imwrite("result3_1.jpg", pano2):
        raise RuntimeError("Failed to save result2.jpg")