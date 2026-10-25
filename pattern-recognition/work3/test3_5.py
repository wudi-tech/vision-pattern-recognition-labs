# Public archive copy. Source mapping and any changes: ../docs/SOURCE_MANIFEST.json
"""
test3_5.py — 原始像素 + 行列统计 + HOG 的 SVM 人脸识别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【特征构成】
  原始像素       361 维   全局强度分布，RBF 核的基础
  行列统计        42 维   水平/垂直方向亮度分布，中心/边缘活跃度（来自 test3_2）
  HOG             81 维   梯度方向直方图，捕捉边缘和轮廓
  ─────────────────────
  合计           484 维

【HOG 原理】
─────────────────────────────────────────────────────────────────────
HOG（Histogram of Oriented Gradients）由 Dalal & Triggs 2005 年提出，
最初用于行人检测，后广泛用于人脸检测，是深度学习之前的检测标配。

核心思想：用"图像中各位置的梯度方向分布"来描述目标形状。
  人脸有固定的轮廓结构（眼眶边缘、鼻梁、嘴角），
  这些结构在梯度图上表现为特定方向的强梯度，
  HOG 能显式捕捉"哪个区域有什么方向的边缘"。

计算步骤（本实现）：

  Step A：计算梯度
    对每个像素用中央差分：
      dx[r,c] = img[r, c+1] - img[r, c-1]   （水平梯度）
      dy[r,c] = img[r+1, c] - img[r-1, c]   （垂直梯度）
    用 edge padding 处理边界，所有 19×19 像素都能算。

  Step B：计算幅值和方向
    magnitude[r,c] = sqrt(dx² + dy²)
    orientation[r,c] = arctan2(dy, dx) % 180°   （无符号方向，0~180°）
    "无符号"：把 180°~360° 折叠到 0°~180°，边缘方向与朝向无关

    ┌────────────────────────────────────────────────────────┐
    │  示例：dx=1, dy=1                                       │
    │    magnitude = sqrt(2) ≈ 1.41                          │
    │    arctan2(1,1) = 45°  → 归入 [40°, 60°) 区间 bin 2   │
    │                                                         │
    │  示例：dx=-1, dy=1                                      │
    │    magnitude = sqrt(2)                                  │
    │    arctan2(1,-1) = 135° → 归入 [120°, 140°) bin 6      │
    │                                                         │
    │  9 个方向 bin，每 bin 覆盖 20°：                        │
    │    bin 0: [  0°,  20°)  → 接近水平边缘                  │
    │    bin 2: [ 40°,  60°)  → 斜向边缘                      │
    │    bin 4: [ 80°, 100°)  → 接近垂直边缘                  │
    └────────────────────────────────────────────────────────┘

  Step C：空间直方图
    把 19×19 的梯度图划分成 3×3=9 个 cell（每 cell ≈ 6×6 像素）。
    对每个 cell 统计 9-bin 方向直方图，以幅值为权重——
    梯度越强的像素对直方图贡献越大。
    对每个 cell 的直方图做 L2 归一化。

  Step D：拼接
    9 cells × 9 bins = 81 维 HOG 特征。

【与 LBP 的区别】
  LBP：用邻域比较编码局部纹理，对同一严格单调灰度变换保序，但丢掉绝对亮度；本实现并非旋转不变
  HOG：用梯度方向编码边缘结构，捕捉轮廓，保留方向信息
  对人脸检测（是不是脸），HOG 的轮廓方向信息比 LBP 的纹理编码更相关。

【其余参数】
  - 网格搜索 5-fold，step=0.25，span=1.0（用户偏好）
  - tie-breaking：并列 CV 准确率时优先选小 C（继承自 test3_4）
  结果输出到 task3_result_5.txt
"""

import os
import sys
import io
import contextlib
import numpy as np
import scipy.io as sio
from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter


def _suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────── 1. 数据加载 ───────────────────────

def _load_mat(path):
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith("_")][0]
    return np.asarray(mat[key])


def load_data(data_dir=DATA_DIR):
    files = {
        "train_data":  os.path.join(data_dir, "train_data.mat"),
        "train_label": os.path.join(data_dir, "train_label.mat"),
        "test_data":   os.path.join(data_dir, "test_data.mat"),
    }
    for name, p in files.items():
        if not os.path.isfile(p):
            sys.exit(f"[错误] 找不到 {p}，请把 {name}.mat 放到 work3/ 下")

    Xtr = _load_mat(files["train_data"]).astype(np.float64)
    ytr = _load_mat(files["train_label"]).astype(np.float64).ravel()
    Xte = _load_mat(files["test_data"]).astype(np.float64)

    if Xtr.shape[0] == 361 and Xtr.shape[1] != 361:
        Xtr = Xtr.T
    if Xte.shape[0] == 361 and Xte.shape[1] != 361:
        Xte = Xte.T

    print(f"[加载] Xtrain={Xtr.shape}, ytrain={ytr.shape}, Xtest={Xte.shape}")
    print(f"[加载] 训练集 +1: {(ytr == 1).sum()},  -1: {(ytr == -1).sum()}")
    return Xtr, ytr, Xte


# ─────────────────────── 2. 特征工程 ───────────────────────

def _spatial_stats(X):
    """
    行列统计特征（来自 test3_2）：
      每行像素和(19) + 每列像素和(19) + 均值(1) + 标准差(1) + 中心9×9和(1) + 边缘和(1)
    返回: (N, 42)
    """
    imgs = X.reshape(-1, 19, 19)
    row_sum = imgs.sum(axis=2)                                   # (N, 19)
    col_sum = imgs.sum(axis=1)                                   # (N, 19)
    mean    = imgs.mean(axis=(1, 2)).reshape(-1, 1)              # (N, 1)
    std     = imgs.std(axis=(1, 2)).reshape(-1, 1)               # (N, 1)
    center  = imgs[:, 5:14, 5:14].sum(axis=(1, 2)).reshape(-1, 1)  # (N, 1)
    border  = (
        imgs[:,  0,  :].sum(axis=1)
      + imgs[:, -1,  :].sum(axis=1)
      + imgs[:,  :,  0].sum(axis=1)
      + imgs[:,  :, -1].sum(axis=1)
    ).reshape(-1, 1)                                             # (N, 1)
    return np.hstack([row_sum, col_sum, mean, std, center, border])  # (N, 42)


def _hog_features(X, n_row=3, n_col=3, n_bins=9):
    """
    HOG 特征提取，全程向量化。

    梯度计算：中央差分，edge padding 处理边界（无需丢弃边缘像素）。
      dx = img[r, c+1] - img[r, c-1]  （右邻 - 左邻）
      dy = img[r+1, c] - img[r-1, c]  （下邻 - 上邻）

    方向量化：
      orientation = arctan2(dy, dx) % 180°，均匀分成 n_bins 个区间
      以 magnitude 为权重累加到对应 bin（梯度越强贡献越大）

    归一化：每个 cell 的直方图做 L2 归一化（加 ε=1e-6 防除零）

    返回: (N, n_row × n_col × n_bins)，默认 3×3×9 = 81 维
    """
    N = X.shape[0]
    imgs = X.reshape(N, 19, 19)

    # ── Step A：中央差分梯度（edge padding 保持 19×19 大小）──────
    # pad 一圈边缘像素，使边界像素也能做中央差分
    pad = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode='edge')  # (N, 21, 21)

    # dx：水平梯度（右邻 - 左邻）
    dx = pad[:, 1:20, 2:21] - pad[:, 1:20, 0:19]              # (N, 19, 19)
    # dy：垂直梯度（下邻 - 上邻）
    dy = pad[:, 2:21, 1:20] - pad[:, 0:19, 1:20]              # (N, 19, 19)

    # ── Step B：幅值 & 无符号方向（0~180°）─────────────────────
    magnitude   = np.sqrt(dx ** 2 + dy ** 2)                   # (N, 19, 19)
    orientation = np.degrees(np.arctan2(dy, dx)) % 180         # (N, 19, 19)

    # 把连续方向映射到离散 bin 编号（0 ~ n_bins-1）
    bin_width = 180.0 / n_bins
    bin_idx   = np.floor(orientation / bin_width).astype(int)
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)                # (N, 19, 19)

    # ── Step C：空间直方图 ─────────────────────────────────────
    row_edges = np.round(np.linspace(0, 19, n_row + 1)).astype(int)
    col_edges = np.round(np.linspace(0, 19, n_col + 1)).astype(int)

    blocks = []
    for i in range(n_row):
        for j in range(n_col):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]

            cell_mag = magnitude[:, r0:r1, c0:c1]  # (N, cell_h, cell_w)
            cell_bin = bin_idx[:, r0:r1, c0:c1]    # (N, cell_h, cell_w)

            # 对每个 bin，把该 cell 内属于该 bin 的像素幅值求和
            hist = np.zeros((N, n_bins))
            for b in range(n_bins):
                hist[:, b] = (cell_mag * (cell_bin == b)).sum(axis=(1, 2))

            # L2 归一化：||hist||₂ + ε
            norm = np.sqrt((hist ** 2).sum(axis=1, keepdims=True) + 1e-6)
            hist = hist / norm

            blocks.append(hist)

    return np.hstack(blocks)  # (N, n_row × n_col × n_bins)


def build_features(X):
    """
    最终特征 = 原始像素(361) + 行列统计(42) + HOG 3×3(81) = 484 维
    在缩放前调用，对原始像素值操作。
    """
    spatial = _spatial_stats(X)          # (N, 42)
    hog     = _hog_features(X)           # (N, 81)
    return np.hstack([X, spatial, hog])  # (N, 484)


# ─────────────────────── 3. 缩放 ───────────────────────

class MinMaxScaler:
    def __init__(self, lo=-1.0, hi=1.0):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
        self.rng  = self.xmax - self.xmin
        self.rng[self.rng == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.xmin) / self.rng * (self.hi - self.lo) + self.lo


# ─────────────────────── 4. 网格搜索 + 交叉验证 ───────────────────────

def cv_accuracy(y, X, C, gamma, nfold=5):
    param = f"-s 0 -t 2 -c {C} -g {gamma} -v {nfold} -q"
    with _suppress_stdout():
        return svm_train(y.tolist(), X.tolist(), param)


def grid_search(y, X,
                C_exps=range(-5, 16, 2),
                g_exps=range(-15, 4, 2),
                nfold=5,
                tag="粗网格"):
    print(f"\n=== {tag}网格搜索 ===")
    print(f"    C 取值  : 2^{{{list(C_exps)}}}")
    print(f"    γ 取值  : 2^{{{list(g_exps)}}}")
    print(f"    交叉验证: {nfold}-fold")

    best   = {"acc": -1, "C": None, "g": None, "ce": None, "ge": None}
    grid   = np.full((len(list(C_exps)), len(list(g_exps))), np.nan)
    C_list = list(C_exps)
    g_list = list(g_exps)

    for i, ce in enumerate(C_list):
        for j, ge in enumerate(g_list):
            C, g = 2.0 ** ce, 2.0 ** ge
            acc  = cv_accuracy(y, X, C, g, nfold)
            grid[i, j] = acc
            flag = ""
            if acc > best["acc"]:
                best.update(acc=acc, C=C, g=g, ce=ce, ge=ge)
                flag = "  <-- best"
            elif acc == best["acc"] and best["C"] is not None:
                # tie-breaking：CV 相同时优先小 C，C 相同时优先小 γ
                if C < best["C"] or (C == best["C"] and g < best["g"]):
                    best.update(acc=acc, C=C, g=g, ce=ce, ge=ge)
                    flag = "  <-- best (tie→小C优先)"
            print(f"    C=2^{ce:+6.2f}={C:10.4g}  γ=2^{ge:+6.2f}={g:10.4g}  acc={acc:6.3f}%{flag}")

    print(f"[{tag}结论] 最佳 C=2^{best['ce']:+.2f}={best['C']}, "
          f"γ=2^{best['ge']:+.2f}={best['g']},  CV准确率={best['acc']:.3f}%")
    return best, grid, C_list, g_list


def fine_grid_around(best, y, X, step=0.25, span=1.0, nfold=5):
    ce, ge = best["ce"], best["ge"]
    C_exps = np.arange(ce - span, ce + span + 1e-9, step)
    g_exps = np.arange(ge - span, ge + span + 1e-9, step)
    return grid_search(y, X, C_exps=C_exps, g_exps=g_exps,
                       nfold=nfold, tag="细")


# ─────────────────────── 5. 训练 & 预测 ───────────────────────

def train_final(y, X, C, gamma):
    prob  = svm_problem(y.tolist(), X.tolist())
    param = svm_parameter(f"-s 0 -t 2 -c {C} -g {gamma} -q")
    with _suppress_stdout():
        return svm_train(prob, param)


def predict(model, X, y_dummy=None):
    if y_dummy is None:
        y_dummy = [0.0] * X.shape[0]
    p_lab, _, _ = svm_predict(y_dummy, X.tolist(), model, "-q")
    return np.asarray(p_lab)


def store_txt(labels, result_txt):
    with open(result_txt, "w") as w:
        for index, lab in enumerate(labels, 1):
            w.write(f"{index} {int(lab)}\n")


# ─────────────────────── 主流程 ───────────────────────

def main():
    # Step 1 —— 加载
    import argparse
    parser = argparse.ArgumentParser(description="Historical face/non-face SVM experiment; supply your own authorized MAT data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/face")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    Xtr, ytr, Xte = load_data(args.data_dir)

    # Step 1.5 —— 构建特征（缩放前对原始像素操作）
    print("[特征] 正在构建原始像素 + 行列统计 + HOG 特征...")
    Xtr = build_features(Xtr)
    Xte = build_features(Xte)
    print(f"[特征] 维数: 361(像素) + 42(行列统计) + 81(HOG 3×3×9) = {Xtr.shape[1]}")

    # Step 2 —— MinMax 缩放
    scaler = MinMaxScaler(-1.0, 1.0).fit(Xtr)
    Xtr_s  = scaler.transform(Xtr)
    Xte_s  = scaler.transform(Xte)
    print(f"[缩放] 训练集范围: [{Xtr_s.min():.3f}, {Xtr_s.max():.3f}]")
    print(f"[缩放] 测试集范围: [{Xte_s.min():.3f}, {Xte_s.max():.3f}]")

    # Step 3+4 —— 粗网格 + 细网格
    best, _, _, _ = grid_search(ytr, Xtr_s)
    best, _, _, _ = fine_grid_around(best, ytr, Xtr_s)

    # Step 5 —— 全训练集重训
    print(f"\n=== 用最佳参数在全训练集上训练 ===")
    print(f"    C={best['C']},  γ={best['g']}")
    model = train_final(ytr, Xtr_s, best["C"], best["g"])

    p_train = predict(model, Xtr_s, ytr.tolist())
    print(f"[训练集精度] {(p_train == ytr).mean() * 100:.3f}%")

    # Step 6 —— 预测 & 保存
    y_pred   = predict(model, Xte_s)
    out_path = os.path.join(args.output_dir, "task3_result_5.txt")
    store_txt(y_pred, out_path)
    print(f"[输出] 共 {len(y_pred)} 个样本，提交格式已保存到 {out_path}")
    print(f"[输出] 预测 +1: {(y_pred == 1).sum()},  -1: {(y_pred == -1).sum()}")


if __name__ == "__main__":
    main()
