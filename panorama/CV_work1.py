import cv2 as cv
import numpy as np
from itertools import combinations
from math import lgamma, log, exp


# 三张待拼接图片。默认顺序为：左图、中间图、右图。
IMAGE_PATHS = ["1.jpg", "2.jpg", "3.jpg"]

# 为了避免原图过大导致运行很慢，默认把最长边缩到 1600。
MAX_SIDE = 1600



def resize_image(img, max_side):
    """将图像最长边缩放到 max_side，保持宽高比例不变。"""
    if max_side is None:
        return img

    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side <= max_side:
        return img
    
    # 缩放比例 = 目标最长边 / 原图最长边。
    scale = max_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)


def calculate_probability(ni, nf, p1=0.6, p0=0.1, prior=1e-6):
    """根据 RANSAC 内点数 ni 和候选匹配数 nf 计算两图匹配概率。"""
    if nf == 0:
        return 0.0

    # 二项分布的对数形式，避免直接计算阶乘造成数值过大。
    log_c = lgamma(nf + 1) - lgamma(ni + 1) - lgamma(nf - ni + 1)
    log_b1 = log_c + ni * log(p1) + (nf - ni) * log(1 - p1)
    log_b0 = log_c + ni * log(p0) + (nf - ni) * log(1 - p0)

    # prior 先验概率：在不知道图像内容前，默认两张图真正匹配的概率很小。
    log_true = log_b1 + log(prior)
    log_false = log_b0 + log(1 - prior)
    x = log_false - log_true

    # 防止 exp(x) 因 x 过大而溢出。
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0

    # 把两个假设的对数概率转换成 0~1 之间的后验概率。
    return 1.0 / (1.0 + exp(x))


def find_transform(img1, img2):
    """计算 img1 到 img2 的单应矩阵，同时返回匹配统计信息。"""
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 创建 SIFT 特征提取器
    sift = cv.SIFT_create()
    # kp1 是关键点列表，des1 是每个关键点对应的 128 维 SIFT 描述子。
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, 0, 0, 0.0

    # 使用 FLANN 匹配。
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(des1, des2, k=2)

    # 最佳匹配要明显优于第二佳匹配。
    good_matches = []
    ratio = 0.75
    for pair in matches:
        if len(pair) == 2:
            m = pair[0]
            n = pair[1]
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # nf：通过初步筛选后的候选匹配数量。
    nf = len(good_matches)
    if nf < 4:
        return None, nf, 0, 0.0

    # src_points 存 img1 中的匹配点坐标；dst_points 存 img2 中对应点坐标。
    src_points = []
    dst_points = []
    for m in good_matches:
        src_points.append(kp1[m.queryIdx].pt)
        dst_points.append(kp2[m.trainIdx].pt)

    src_points = np.float32(src_points).reshape(-1, 1, 2)
    dst_points = np.float32(dst_points).reshape(-1, 1, 2)

    # RANSAC 可以剔除错误匹配，mask 中的 1 表示内点。
    H, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    if H is None or mask is None:
        return None, nf, 0, 0.0

    # 内点数量 ni。
    ni = int(mask.sum())
    probability = calculate_probability(ni, nf)
    return H, nf, ni, probability


def crop_black_border(img, valid_mask):
    """根据有效区域 mask 裁掉四周黑边。"""

    # 真实图像内容
    rows, cols = np.where(valid_mask > 0)

    if len(rows) == 0 or len(cols) == 0:
        return img

    # 找到所有有效像素形成的最小外接矩形。
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(cols.min())
    right = int(cols.max()) + 1

    # 向内收缩边界。每次裁掉“黑边最多”的一侧，
    threshold = 0.995
    # 持续检查四条边是否还含有明显黑边。
    while top < bottom - 1 and left < right - 1:
        top_ratio = valid_mask[top, left:right].mean()
        bottom_ratio = valid_mask[bottom - 1, left:right].mean()
        left_ratio = valid_mask[top:bottom, left].mean()
        right_ratio = valid_mask[top:bottom, right - 1].mean()

        # 找出有效像素比例最低的一条边。
        min_ratio = min(top_ratio, bottom_ratio, left_ratio, right_ratio)
        if min_ratio >= threshold:
            break

        # 哪条边的有效比例最低，就把哪条边向内收缩一格。
        if min_ratio == top_ratio:
            top += 1
        elif min_ratio == bottom_ratio:
            bottom -= 1
        elif min_ratio == left_ratio:
            left += 1
        else:
            right -= 1

    return img[top:bottom, left:right]


def stitch_three_images(images):
    """以第二张图为参考图，将三张图拼接成全景图。"""
    img_a = images[0]
    img_b = images[1]
    img_c = images[2]

    # 计算 A 到 B , C 到 B的单应矩阵。
    H_ab, nf_ab, ni_ab, p_ab = find_transform(img_a, img_b)
    H_cb, nf_cb, ni_cb, p_cb = find_transform(img_c, img_b)

    if H_ab is None or H_cb is None:
        raise ValueError("图像重叠不足，无法完成三图拼接。")

    transforms = [H_ab, np.eye(3), H_cb]

    # 计算三张图变换后的四个角点，用来确定全景图画布大小。
    all_corners = []
    for i in range(3):
        img = images[i]
        H = transforms[i]
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        new_corners = cv.perspectiveTransform(corners, H)
        all_corners.append(new_corners)

    all_corners = np.vstack(all_corners)
    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # 如果坐标有负数，就整体平移到正坐标区域。
    move = np.array([[1, 0, -xmin],
                     [0, 1, -ymin],
                     [0, 0, 1]], dtype=np.float64)

    out_w = xmax - xmin
    out_h = ymax - ymin

    color_sum = np.zeros((out_h, out_w, 3), np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), np.float32)

    for i in range(3):
        img = images[i]
        H = transforms[i]
        final_H = np.dot(move, H)

        warped = cv.warpPerspective(img, final_H, (out_w, out_h))

        # 单独变换一张白色 mask，用来判断哪些像素来自原图。
        src_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv.warpPerspective(src_mask, final_H, (out_w, out_h),
                                         flags=cv.INTER_NEAREST)
        valid = (warped_mask > 0).astype(np.uint8)

        # 距离变换产生羽化权重：越靠近图像中心权重越大，边缘权重越小。
        weight = cv.distanceTransform(valid, cv.DIST_L2, 3)
        if weight.max() > 0:
            weight = weight / weight.max()

        weight = weight.reshape(out_h, out_w, 1).astype(np.float32)
        color_sum = color_sum + warped.astype(np.float32) * weight
        weight_sum = weight_sum + weight

    panorama_full = color_sum / np.maximum(weight_sum, 1e-6)
    panorama_full = np.clip(panorama_full, 0, 255).astype(np.uint8)

    valid_mask = (weight_sum[:, :, 0] > 1e-6).astype(np.uint8)
    panorama_crop = crop_black_border(panorama_full, valid_mask)

    pair_results = [
        ("A-B", nf_ab, ni_ab, p_ab),
        ("B-C", nf_cb, ni_cb, p_cb)
    ]

    return panorama_crop, panorama_full, pair_results


def main():
    images = []
    for path in IMAGE_PATHS:
        img = cv.imread(path)
        if img is None:
            raise FileNotFoundError("无法读取图片：" + path)
        img = resize_image(img, MAX_SIDE)
        images.append(img)

    # 计算每两张图之间的匹配概率。
    print("三对图像的匹配概率：")
    for pair in combinations(range(3), 2):
        i = pair[0]
        j = pair[1]
        H, nf, ni, prob = find_transform(images[i], images[j])
        print(IMAGE_PATHS[i], "-", IMAGE_PATHS[j],
              "nf =", nf, "ni =", ni, "P =", format(prob, ".6f"))

    result_crop, result_full, used_pairs = stitch_three_images(images)

    cv.imwrite("result_crop.jpg", result_crop)
    cv.imwrite("result_full.jpg", result_full)

    print("\n实际用于拼接的相邻图像对：")
    for item in used_pairs:
        name, nf, ni, prob = item
        print(name, "nf =", nf, "ni =", ni, "P =", format(prob, ".6f"))

    print("\n已保存：result_crop.jpg 和 result_full.jpg")


if __name__ == "__main__":
    main()
