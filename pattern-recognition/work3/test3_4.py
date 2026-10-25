# Public archive copy. Source mapping and any changes: ../docs/SOURCE_MANIFEST.json
"""
test3_4.py — 原始像素 + LBP 拼接特征的 SVM 人脸识别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【相比 test3_LBP.py 的改动原因】
  历史用户记录中统计特征为 0.860、LBP-only 为 0.812；协议与测试标签未保留，不能据此确定因果。
  当时提出的解释假设（未做受控消融验证）：
    1. 19×19 图像太小，3×3 cell 每格只有 36 像素，59-bin 直方图极度稀疏
    2. LBP 丢弃了绝对亮度信息，而人脸检测（非识别）恰好依赖整体形状/亮度分布
    3. 替换而非补充，损失了原始像素本身携带的判别信息

【本文件策略：拼接而非替换】
  特征 = 原始像素（361 维）
        + LBP 空间直方图，2×2 cell（236 维）
        ─────────────────────────────────
        合计 597 维

  为什么改用 2×2 cell（而不是 test3_LBP 的 3×3）？
    2×2 cell 下每格约 72 像素（17×17 / 4 ≈ 72），
    填 59 个 bin 时每 bin 平均 ~1.2 个像素，
    比 3×3 的 ~0.6 更可靠，直方图噪声更小。

  原始像素保留全局形状信息，LBP 补充光照鲁棒的纹理信息，两者互补。

【历史参数（以函数默认值为准）】
  - 交叉验证 5-fold
  - 细网格 step=0.25, span=1.0

结果输出到 task3_result_4.txt。
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


# ─────────────────────── 2. LBP 特征提取 ───────────────────────
# LBP 原理详见 test3_LBP.py 注释，这里只保留必要的实现说明。

# 8 邻居偏移，顺时针从正上方 bit0 开始
_NEIGHBOR_OFFSETS = [
    (-1,  0),  # bit 0：正上
    (-1, +1),  # bit 1：右上
    ( 0, +1),  # bit 2：正右
    (+1, +1),  # bit 3：右下
    (+1,  0),  # bit 4：正下
    (+1, -1),  # bit 5：左下
    ( 0, -1),  # bit 6：正左
    (-1, -1),  # bit 7：左上
]

_UNIFORM_LUT_CACHE = None


def _make_uniform_lut():
    """LBP 码（0~255）→ 均匀 bin（0~57）或非均匀 bin（58），共 59 bins。"""
    global _UNIFORM_LUT_CACHE
    if _UNIFORM_LUT_CACHE is not None:
        return _UNIFORM_LUT_CACHE

    lut = np.full(256, 58, dtype=np.int32)
    uniform_idx = 0
    for code in range(256):
        bits = np.array([(code >> i) & 1 for i in range(8)], dtype=np.int32)
        if int(np.sum(bits != np.roll(bits, -1))) <= 2:   # 循环跳变次数 ≤ 2
            lut[code] = uniform_idx
            uniform_idx += 1

    _UNIFORM_LUT_CACHE = lut
    return lut


def _compute_lbp_codes(imgs):
    """
    对 (N, 19, 19) 图像批量计算内部 17×17 像素的 LBP 码。

    向量化：对每个方向取邻居切片，与中心比较后按位累加。
    返回: (N, 17, 17)，uint8，值域 0~255。
    """
    center = imgs[:, 1:18, 1:18]                          # (N, 17, 17)
    lbp = np.zeros((imgs.shape[0], 17, 17), dtype=np.uint8)
    for bit, (dr, dc) in enumerate(_NEIGHBOR_OFFSETS):
        neighbor = imgs[:, 1 + dr : 18 + dr, 1 + dc : 18 + dc]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def extract_lbp_features(X, n_row=2, n_col=2):
    """
    从原始像素提取 LBP 空间直方图特征。

    流程：原始像素 → LBP 码图(17×17) → 均匀映射(0~58)
          → n_row×n_col cell 划分 → 每 cell 59-bin 直方图(L1归一化)
          → 拼接所有 cell → (N, n_row×n_col×59)

    默认 2×2=4 cells，每 cell 约 72 像素，直方图比 3×3 更可靠。
    """
    N = X.shape[0]
    lut = _make_uniform_lut()
    lbp_uniform = lut[_compute_lbp_codes(X.reshape(N, 19, 19))]  # (N,17,17)

    row_edges = np.round(np.linspace(0, 17, n_row + 1)).astype(int)
    col_edges = np.round(np.linspace(0, 17, n_col + 1)).astype(int)

    blocks = []
    for i in range(n_row):
        for j in range(n_col):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]
            cell = lbp_uniform[:, r0:r1, c0:c1].reshape(N, -1)   # (N, pixels)

            # 59-bin 频率直方图，L1 归一化
            hist = np.vstack([
                np.bincount(cell[n], minlength=59) for n in range(N)
            ]).astype(np.float64)
            hist /= (r1 - r0) * (c1 - c0)    # 变为频率（÷像素数）

            blocks.append(hist)

    return np.hstack(blocks)                  # (N, n_row×n_col×59)


def build_features(X):
    """
    最终特征 = 原始像素（361）+ LBP 直方图（2×2×59=236）= 597 维。
    在缩放前调用，对原始像素值操作。
    """
    lbp = extract_lbp_features(X, n_row=2, n_col=2)   # (N, 236)
    return np.hstack([X, lbp])                          # (N, 597)


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


# ─────────────────────── 4. 网格搜索 + 交叉验证（5-fold）───────────────────────

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
                # 准确率严格提升，无条件更新
                best.update(acc=acc, C=C, g=g, ce=ce, ge=ge)
                flag = "  <-- best"
            elif acc == best["acc"] and best["C"] is not None:
                # 准确率并列时的 tie-breaking 规则：
                #   优先选更小的 C（正则化更强，决策边界更平滑，泛化更好）
                #   C 相同时选更小的 gamma（RBF 核更宽，决策边界更平滑）
                # 理论依据：SVM 结构风险最小化——间隔越大，VC 维越低，泛化越好
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

    # Step 1.5 —— 拼接 LBP 特征（在缩放前对原始像素操作）
    #   原始像素携带全局形状，LBP 补充光照鲁棒的局部纹理，两者互补
    print("[特征] 正在构建原始像素 + LBP 拼接特征...")
    Xtr = build_features(Xtr)   # (N_train, 597)
    Xte = build_features(Xte)   # (N_test,  597)
    print(f"[特征] 维数: 361(原始像素) + 4×59(LBP 2×2) = {Xtr.shape[1]}")

    # Step 2 —— MinMax 缩放（在拼接后的全特征上 fit）
    scaler = MinMaxScaler(-1.0, 1.0).fit(Xtr)
    Xtr_s  = scaler.transform(Xtr)
    Xte_s  = scaler.transform(Xte)
    print(f"[缩放] 训练集范围: [{Xtr_s.min():.3f}, {Xtr_s.max():.3f}]")
    print(f"[缩放] 测试集范围: [{Xte_s.min():.3f}, {Xte_s.max():.3f}]")

    # Step 3+4 —— 粗网格 + 细网格（5-fold）
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
    out_path = os.path.join(args.output_dir, "task3_result_4.txt")
    store_txt(y_pred, out_path)
    print(f"[输出] 共 {len(y_pred)} 个样本，提交格式已保存到 {out_path}")
    print(f"[输出] 预测 +1: {(y_pred == 1).sum()},  -1: {(y_pred == -1).sum()}")


if __name__ == "__main__":
    main()
