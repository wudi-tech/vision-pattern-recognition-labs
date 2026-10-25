# ============================================================
# 带注释学习版
# 说明：本文件只在原代码基础上增加中文注释，没有改动任何可执行语句。
# 代码整体功能：读取 3 张图片，通过 SIFT + FLANN + RANSAC 求单应矩阵，
#              再把三张图统一变换到同一画布中，融合并裁剪黑边，得到全景图。
# ============================================================

# cv2 是 OpenCV 库，用于图像读取、特征提取、图像变换、保存结果等操作。
import cv2 as cv
# numpy 用于矩阵、数组和坐标计算。
import numpy as np
# combinations 用于生成三张图片之间的两两组合，例如 (0,1)、(0,2)、(1,2)。
from itertools import combinations
# lgamma、log、exp 用于后面计算匹配概率，使用对数形式可以避免数值溢出。
from math import lgamma, log, exp



# --------------------------
# 全局参数设置
# --------------------------
# 三张待拼接图片。默认顺序为：左图、中间图、右图。
IMAGE_PATHS = ["1.jpg", "2.jpg", "3.jpg"]

# 为了避免原图过大导致运行很慢，默认把最长边缩到 1600。
MAX_SIDE = 1600




# --------------------------
# 工具函数 1：缩放图像
# --------------------------
def resize_image(img, max_side):
    """将图像最长边缩放到 max_side，保持宽高比例不变。"""
    # 如果 max_side 传入 None，表示不需要缩放，直接返回原图。
    if max_side is None:
        return img

    # img.shape[:2] 只取图像的高和宽；彩色图像的完整 shape 一般是 (高, 宽, 通道数)。
    h, w = img.shape[:2]
    # 找到宽和高中较长的一边，判断是否超过设定的最大边长。
    long_side = max(h, w)

    # 如果图片本来就不大，就不做 resize，避免不必要的图像质量损失。
    if long_side <= max_side:
        return img

    # 缩放比例 = 目标最长边 / 原图最长边；宽和高都乘这个比例，才能保持原图比例不变。
    scale = max_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    # INTER_AREA 通常适合图像缩小，效果比较稳定。
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)



# --------------------------
# 工具函数 2：根据匹配质量估计两张图是否真正匹配
# --------------------------
def calculate_probability(ni, nf, p1=0.6, p0=0.1, prior=1e-6):
    """根据 RANSAC 内点数 ni 和候选匹配数 nf 计算两图匹配概率。"""
    # nf 表示通过 ratio test 后留下的候选匹配数；如果为 0，说明没有可靠匹配。
    if nf == 0:
        return 0.0

    # 二项分布的对数形式，避免直接计算阶乘造成数值过大。
    # log_c 是二项分布中的组合数 C(nf, ni) 的对数。
    # 使用 lgamma 是因为 lgamma(n+1) 等价于 log(n!)，可避免直接计算大阶乘。
    log_c = lgamma(nf + 1) - lgamma(ni + 1) - lgamma(nf - ni + 1)
    # log_b1：假设两张图确实匹配时，观察到 ni 个内点的概率对数。
    log_b1 = log_c + ni * log(p1) + (nf - ni) * log(1 - p1)
    # log_b0：假设两张图不匹配时，观察到 ni 个内点的概率对数。
    log_b0 = log_c + ni * log(p0) + (nf - ni) * log(1 - p0)

    # prior 是先验概率：在不知道图像内容前，默认两张图真正匹配的概率很小。
    log_true = log_b1 + log(prior)
    log_false = log_b0 + log(1 - prior)
    # x 越大，说明“不匹配”的可能性越强；x 越小，说明“匹配”的可能性越强。
    x = log_false - log_true

    # 防止 exp(x) 溢出。
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0

    # 这里相当于把两个假设的对数概率转换成 0~1 之间的后验概率。
    return 1.0 / (1.0 + exp(x))



# --------------------------
# 核心函数 1：计算两张图之间的单应矩阵 H
# --------------------------
def find_transform(img1, img2):
    """计算 img1 到 img2 的单应矩阵，同时返回匹配统计信息。"""
    # SIFT 通常在灰度图上提取特征点，因此先把 BGR 彩色图转成灰度图。
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 创建 SIFT 特征提取器：用于找到图像中的关键点，并计算每个关键点的描述子。
    sift = cv.SIFT_create()
    # kp1 是关键点列表，des1 是每个关键点对应的 128 维 SIFT 描述子。
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 如果某张图没有检测到特征描述子，就无法进行匹配。
    if des1 is None or des2 is None:
        return None, 0, 0, 0.0

    # SIFT 描述子是浮点特征，适合使用 FLANN 匹配。
    # algorithm=1 表示使用 KD-Tree；trees=5 表示建立 5 棵树，通常用于 SIFT 这类浮点描述子。
    index_params = dict(algorithm=1, trees=5)
    # checks 越大，搜索越充分但越慢；50 是常用的折中值。
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)

    # 对 img1 的每个描述子，在 img2 中找最近的 2 个候选匹配点。
    # 之所以找 2 个，是为了后面做 Lowe ratio test。
    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe ratio test：最佳匹配要明显优于第二佳匹配。
    # good_matches 用来保存通过筛选的可靠匹配点。
    good_matches = []
    # ratio 越小，筛选越严格；0.75 是 SIFT 匹配中常用经验值。
    ratio = 0.75
    # pair 通常包含两个匹配结果：最近邻 m 和次近邻 n。
    for pair in matches:
        if len(pair) == 2:
            m = pair[0]
            n = pair[1]
            # 如果最近邻距离明显小于次近邻，说明这个匹配比较有辨识度，保留下来。
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # nf：通过初步筛选后的候选匹配数量。
    nf = len(good_matches)
    # 单应矩阵至少需要 4 对点才能求解，所以 nf < 4 时直接失败。
    if nf < 4:
        return None, nf, 0, 0.0

    # src_points 存 img1 中的匹配点坐标；dst_points 存 img2 中对应点坐标。
    src_points = []
    dst_points = []
    # queryIdx 是匹配点在 img1 关键点列表中的索引；trainIdx 是在 img2 关键点列表中的索引。
    for m in good_matches:
        src_points.append(kp1[m.queryIdx].pt)
        dst_points.append(kp2[m.trainIdx].pt)

    # OpenCV 的 findHomography 需要点坐标格式为 (N, 1, 2)，所以这里 reshape。
    src_points = np.float32(src_points).reshape(-1, 1, 2)
    dst_points = np.float32(dst_points).reshape(-1, 1, 2)

    # RANSAC 可以剔除错误匹配，mask 中的 1 表示内点。
    # cv.RANSAC 会在含有错误匹配的情况下估计 H，并返回哪些匹配是内点。
    # 5.0 表示 RANSAC 重投影误差阈值，单位是像素。
    H, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    if H is None or mask is None:
        return None, nf, 0, 0.0

    # mask 中内点位置为 1，外点位置为 0；求和就是内点数量 ni。
    ni = int(mask.sum())
    # 根据内点数量 ni 和候选匹配数量 nf，估计这两张图属于同一场景的概率。
    probability = calculate_probability(ni, nf)
    # 返回值依次是：单应矩阵 H、候选匹配数 nf、内点数 ni、匹配概率 probability。
    return H, nf, ni, probability



# --------------------------
# 工具函数 3：裁剪拼接结果四周的黑边
# --------------------------
def crop_black_border(img, valid_mask):
    """根据有效区域 mask 裁掉四周黑边。"""
    # valid_mask 中大于 0 的位置代表有真实图像内容，不是黑色空白区域。
    rows, cols = np.where(valid_mask > 0)

    # 如果 mask 中完全没有有效像素，就无法裁剪，直接返回原图。
    if len(rows) == 0 or len(cols) == 0:
        return img

    # 先找到所有有效像素形成的最小外接矩形。
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(cols.min())
    right = int(cols.max()) + 1

    # 继续向内收缩边界。每次裁掉“黑边最多”的一侧，
    # 比固定先裁上边更容易保留主要画面。
    # threshold 接近 1，表示边界行/列几乎全是有效像素时才停止继续裁剪。
    threshold = 0.995
    # 只要矩形区域还足够大，就持续检查四条边是否还含有明显黑边。
    while top < bottom - 1 and left < right - 1:
        # 计算当前上边界这一行的有效像素比例。
        top_ratio = valid_mask[top, left:right].mean()
        # 计算当前下边界这一行的有效像素比例。
        bottom_ratio = valid_mask[bottom - 1, left:right].mean()
        # 计算当前左边界这一列的有效像素比例。
        left_ratio = valid_mask[top:bottom, left].mean()
        # 计算当前右边界这一列的有效像素比例。
        right_ratio = valid_mask[top:bottom, right - 1].mean()

        # 找出有效像素比例最低的一条边，它最可能是黑边最严重的一侧。
        min_ratio = min(top_ratio, bottom_ratio, left_ratio, right_ratio)
        # 如果四条边都足够“干净”，说明基本没有黑边，可以停止。
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

    # 按最终得到的上下左右边界切出裁剪后的图像。
    return img[top:bottom, left:right]



# --------------------------
# 核心函数 2：三图拼接
# --------------------------
def stitch_three_images(images):
    """以第二张图为参考图，将三张图拼接成全景图。"""
    # 为了便于理解，把三张图分别命名为 A、B、C，其中 B 是中间参考图。
    img_a = images[0]
    img_b = images[1]
    img_c = images[2]

    # 计算 A 到 B 的单应矩阵，即把左图 A 变换到中间图 B 的坐标系。
    H_ab, nf_ab, ni_ab, p_ab = find_transform(img_a, img_b)
    # 计算 C 到 B 的单应矩阵，即把右图 C 变换到中间图 B 的坐标系。
    H_cb, nf_cb, ni_cb, p_cb = find_transform(img_c, img_b)

    # 只要 A-B 或 C-B 中任意一对无法求出 H，三图拼接就无法继续。
    if H_ab is None or H_cb is None:
        raise ValueError("图像重叠不足，无法完成三图拼接。")

    # transforms 中三个矩阵分别对应 A、B、C。
    # B 是参考图，所以用单位矩阵 np.eye(3)，表示它不需要变换。
    transforms = [H_ab, np.eye(3), H_cb]

    # 计算三张图变换后的四个角点，用来确定全景图画布大小。
    # all_corners 用于收集三张图变换后的角点坐标。
    all_corners = []
    # 依次处理 A、B、C 三张图。
    for i in range(3):
        img = images[i]
        H = transforms[i]
        h, w = img.shape[:2]
        # 原图四个角点坐标：左上、右上、右下、左下。
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        # 使用对应的单应矩阵 H，把角点投影到全景图的统一坐标系中。
        new_corners = cv.perspectiveTransform(corners, H)
        all_corners.append(new_corners)

    # 把三张图的所有角点堆叠成一个数组，方便统一求最小/最大坐标。
    all_corners = np.vstack(all_corners)
    # xmin、ymin 是所有变换后角点中的最小 x/y 坐标。
    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    # xmax、ymax 是所有变换后角点中的最大 x/y 坐标。
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # 如果坐标有负数，就整体平移到正坐标区域。
    move = np.array([[1, 0, -xmin],
                     [0, 1, -ymin],
                     [0, 0, 1]], dtype=np.float64)

    # 全景图画布宽度和高度由变换后的最大/最小坐标差决定。
    out_w = xmax - xmin
    out_h = ymax - ymin

    # color_sum 用于累加每张变换后图像的像素值。
    color_sum = np.zeros((out_h, out_w, 3), np.float32)
    # weight_sum 用于累加每个像素位置的权重，最后做加权平均。
    weight_sum = np.zeros((out_h, out_w, 1), np.float32)

    # 将三张图逐张投影到最终画布上，并进行加权融合。
    for i in range(3):
        img = images[i]
        H = transforms[i]
        # final_H = 平移矩阵 move × 当前图像到参考图的变换矩阵 H。
        # 这样可以保证变换后的坐标全部落在正数画布范围内。
        final_H = np.dot(move, H)

        # warpPerspective 根据 final_H 对图像做透视变换，并放到统一大小的画布上。
        warped = cv.warpPerspective(img, final_H, (out_w, out_h))

        # 单独变换一张白色 mask，用来判断哪些像素来自原图。
        src_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv.warpPerspective(src_mask, final_H, (out_w, out_h),
                                         flags=cv.INTER_NEAREST)
        # valid 是 0/1 mask：1 表示该位置有原图投影过来的有效内容。
        valid = (warped_mask > 0).astype(np.uint8)

        # 距离变换产生羽化权重：越靠近图像中心权重越大，边缘权重越小。
        weight = cv.distanceTransform(valid, cv.DIST_L2, 3)
        if weight.max() > 0:
            weight = weight / weight.max()

        # 将二维权重图 reshape 成 (高, 宽, 1)，方便与三通道彩色图像相乘。
        weight = weight.reshape(out_h, out_w, 1).astype(np.float32)
        # 将当前图像按权重加入总颜色累加器。
        color_sum = color_sum + warped.astype(np.float32) * weight
        # 累加权重，记录每个像素位置总共被多少有效图像贡献。
        weight_sum = weight_sum + weight

    # 对重叠区域做加权平均；np.maximum 防止除以 0。
    panorama_full = color_sum / np.maximum(weight_sum, 1e-6)
    # 把像素值限制在 0~255，并转回 uint8，变成正常可保存的图像格式。
    panorama_full = np.clip(panorama_full, 0, 255).astype(np.uint8)

    # 只要某个位置权重大于 0，说明它至少来自一张原图，是有效区域。
    valid_mask = (weight_sum[:, :, 0] > 1e-6).astype(np.uint8)
    # 根据有效区域 mask 裁掉四周黑边，得到更适合提交/查看的结果。
    panorama_crop = crop_black_border(panorama_full, valid_mask)

    # 保存实际用于拼接的两组相邻图片匹配统计，方便最后打印检查。
    pair_results = [
        ("A-B", nf_ab, ni_ab, p_ab),
        ("B-C", nf_cb, ni_cb, p_cb)
    ]

    # 同时返回裁剪版、未裁剪完整版、匹配统计结果。
    return panorama_crop, panorama_full, pair_results



# --------------------------
# 主函数：程序入口
# --------------------------
def main():
    # images 用来保存读取并缩放后的三张图片。
    images = []
    # 依次读取 IMAGE_PATHS 中指定的图片文件。
    for path in IMAGE_PATHS:
        # cv.imread 默认读取为 BGR 格式；如果路径错误或文件不存在，会返回 None。
        img = cv.imread(path)
        # 检查图片是否成功读取。
        if img is None:
            raise FileNotFoundError("无法读取图片：" + path)
        # 如果图片太大，就缩放到 MAX_SIDE 限制范围内。
        img = resize_image(img, MAX_SIDE)
        # 把处理后的图片加入列表，后面统一用于匹配和拼接。
        images.append(img)

    # 计算每两张图之间的匹配概率。
    print("三对图像的匹配概率：")
    # combinations(range(3), 2) 会生成三张图的全部两两组合。
    for pair in combinations(range(3), 2):
        i = pair[0]
        j = pair[1]
        # 对每一对图像都计算匹配情况，用于判断哪两张图更可能相邻。
        H, nf, ni, prob = find_transform(images[i], images[j])
        print(IMAGE_PATHS[i], "-", IMAGE_PATHS[j],
              "nf =", nf, "ni =", ni, "P =", format(prob, ".6f"))

    # 真正执行三图拼接，返回裁剪版、完整版以及实际使用的相邻匹配结果。
    result_crop, result_full, used_pairs = stitch_three_images(images)

    # 保存裁剪后的全景图，通常这是最终更美观的结果。
    cv.imwrite("submit_result_crop.jpg", result_crop)
    # 保存未裁剪的完整画布，方便检查原始拼接范围和黑边情况。
    cv.imwrite("submit_result_full.jpg", result_full)

    print("\n实际用于拼接的相邻图像对：")
    # 打印 A-B 和 B-C 两组实际参与拼接的匹配质量。
    for item in used_pairs:
        name, nf, ni, prob = item
        print(name, "nf =", nf, "ni =", ni, "P =", format(prob, ".6f"))

    print("\n已保存：submit_result_crop.jpg 和 submit_result_full.jpg")



# 只有直接运行这个 .py 文件时，下面的 main() 才会执行。
# 如果这个文件被其他 Python 文件 import，main() 不会自动运行。
if __name__ == "__main__":
    main()
