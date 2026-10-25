# Public archive copy. Source mapping and any changes: ../docs/SOURCE_MANIFEST.json
"""
test3_LBP.py — 基于 LBP (Local Binary Pattern) 特征的 SVM 人脸识别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【一、什么是 LBP？】
─────────────────────────────────────────────────────────────────────
LBP 由 Ojala 等人于 1994/2002 年提出，用于描述图像纹理。
2006 年 Ahonen 等人将其用于人脸识别，成为深度学习之前的业界标配。

核心思想：不看像素的绝对亮度，而是看"每个像素与它邻居的大小关系"。
优点：对同一严格单调灰度变换保序（不含截断/量化改变比较的情况）——整体变亮或变暗不影响比较结果。

【二、LBP 编码的计算过程（P=8, R=1，最基本版本）】
─────────────────────────────────────────────────────────────────────
对图像中的每一个（非边界）像素，取它 3×3 邻域内的 8 个邻居，
按顺时针方向逐个与中心像素比较：

    邻居 >= 中心像素 → 该位记为 1
    邻居  < 中心像素 → 该位记为 0

把 8 个比较结果按位拼成一个 8 位二进制数（0~255），即"LBP 码"。

    邻居排布（顺时针，bit 0 = 正上方）：
        bit7  bit0  bit1
        bit6   C    bit2
        bit5  bit4  bit3

    ┌──────────────────────────────────────────────────┐
    │  示例：中心像素 C = 90                            │
    │                                                   │
    │  邻居值:  80   60                                 │
    │          120  [90]  50    ← 中心                  │
    │          140   100  130                           │
    │                                                   │
    │  顺时针从 bit0(正上) 开始比较：                   │
    │    bit0: 正上  80 <  90  → 0                      │
    │    bit1: 右上  60 <  90  → 0                      │
    │    bit2: 右    50 <  90  → 0                      │
    │    bit3: 右下 130 >= 90  → 1                      │
    │    bit4: 下   100 >= 90  → 1                      │
    │    bit5: 左下 140 >= 90  → 1                      │
    │    bit6: 左   120 >= 90  → 1                      │
    │    bit7: 左上  80 <  90  → 0                      │
    │                                                   │
    │  LBP 码 = 0b01111000 = 120                        │
    └──────────────────────────────────────────────────┘

【三、Uniform LBP（均匀模式）——把 256 个 bin 压缩成 59 个】
─────────────────────────────────────────────────────────────────────
256 种 LBP 码中，只有"均匀模式"在自然图像中频繁出现。
均匀模式定义：8 位编码循环读取时，0→1 和 1→0 的跳变次数 ≤ 2。

    均匀（跳变次数 ≤ 2），代表有意义的结构：
        00000000  (0次跳变) → 平坦暗区域
        11111111  (0次跳变) → 平坦亮区域
        00001111  (2次跳变) → 边缘（暗到亮）
        00111110  (2次跳变) → 亮斑 / 拐角

    非均匀（跳变次数 ≥ 4），多为噪声：
        10101010  (8次跳变) → 随机噪声
        10011010  (6次跳变) → 随机噪声

数学上恰好有 58 种均匀模式（验证：1 + 8×7 + 1 = 58 种单连通弧）。
因此用 59 个 bin（58 种均匀各一个 + 1 个"非均匀总类"）覆盖自然图像约 90% 的 LBP 码。

【四、空间直方图（Spatial Histogram）——保留位置信息】
─────────────────────────────────────────────────────────────────────
人脸有固定空间结构：眼睛在上方、鼻子居中、嘴在下方。
如果把整张图所有像素的 LBP 放进一个直方图，位置信息就丢失了。

解决方案（Ahonen 2006）：
  1. 把图像划分成 n_row × n_col 个小格（cell）
  2. 对每个 cell 单独计算 59-bin 均匀 LBP 直方图，并 L1 归一化
  3. 把所有 cell 的直方图拼接成最终特征向量

    ┌─────────────────────────────────────────────────────┐
    │  19×19 图像 → 内部 17×17 LBP 图 → 3×3 cell 划分    │
    │                                                      │
    │  ┌───┬───┬───┐                                       │
    │  │ 发 │ 额 │ 发 │  ← 每个 cell ≈ 5~6×5~6 像素        │
    │  ├───┼───┼───┤    每 cell 59 bins                    │
    │  │ 眼 │ 鼻 │ 眼 │                                    │
    │  ├───┼───┼───┤    合计 9 × 59 = 531 维               │
    │  │ 脸 │ 嘴 │ 脸 │                                    │
    │  └───┴───┴───┘                                       │
    └─────────────────────────────────────────────────────┘

    相比原始 361 维像素：
      ✓ 对同一严格单调灰度变换保序（不含截断/量化改变比较的情况）
      ✓ 保留了"哪个区域有什么纹理"的空间布局信息
      ✓ 捕捉边缘方向、纹理粗糙度等面部结构特征

【本文件方法变化汇总（不代表准确率改善）】
─────────────────────────────────────────────────────────────────────
  变化 1：用 LBP 空间直方图（531 维）替换原始像素（361 维）
  变化 2：交叉验证 10-fold（继承自 test3_2.py）
  其余流程与 test3_1.py 完全相同
  结果输出到 task3_result_LBP.txt
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

# 8 个邻居相对于中心像素的偏移 (行偏移, 列偏移)
# 顺序：顺时针，从正上方 bit0 开始
#
#    bit7  bit0  bit1
#    bit6   C    bit2
#    bit5  bit4  bit3
#
_NEIGHBOR_OFFSETS = [
    (-1,  0),   # bit 0：正上
    (-1, +1),   # bit 1：右上
    ( 0, +1),   # bit 2：正右
    (+1, +1),   # bit 3：右下
    (+1,  0),   # bit 4：正下
    (+1, -1),   # bit 5：左下
    ( 0, -1),   # bit 6：正左
    (-1, -1),   # bit 7：左上
]

# 模块级缓存，避免重复构建
_UNIFORM_LUT_CACHE = None


def _make_uniform_lut():
    """
    构建 256 → 59 的均匀 LBP 查找表（只构建一次）。

    遍历所有 256 种 LBP 码，统计其 8 位循环跳变次数：
      - 跳变次数 ≤ 2 → 均匀模式，分配独立 bin（0~57，共 58 个）
      - 跳变次数 > 2 → 非均匀模式，统一归入 bin 58

    返回: shape (256,) 的 int32 数组
          code → bin 编号 (0~57 均匀 | 58 非均匀)
    """
    global _UNIFORM_LUT_CACHE
    if _UNIFORM_LUT_CACHE is not None:
        return _UNIFORM_LUT_CACHE

    lut = np.full(256, 58, dtype=np.int32)   # 默认全部是非均匀 bin 58
    uniform_idx = 0

    for code in range(256):
        # 把 code 拆成 8 个 bit，bit[0] 对应 LSB（bit0 = 正上方邻居）
        bits = np.array([(code >> i) & 1 for i in range(8)], dtype=np.int32)

        # 循环跳变次数：把 bits 首尾相接成一个圆环，数 0→1 或 1→0 的变化次数
        # np.roll(bits, -1) 把数组左移一位，相当于取"下一个"邻居位
        transitions = int(np.sum(bits != np.roll(bits, -1)))

        if transitions <= 2:          # 均匀模式
            lut[code] = uniform_idx
            uniform_idx += 1          # 每种均匀模式独占一个 bin

    # uniform_idx 最终 == 58，验证了恰好有 58 种均匀模式
    _UNIFORM_LUT_CACHE = lut
    return lut


def _compute_lbp_codes(imgs):
    """
    对一批 19×19 图像批量计算每个内部像素的原始 LBP 码（0~255）。

    为什么只处理"内部像素"？
      边界像素（第 0/18 行或第 0/18 列）缺少完整的 8 邻域，无法正常编码。
      因此只处理行列均在 [1, 17] 范围内的 17×17 = 289 个内部像素。

    向量化实现思路：
      - center = imgs[:, 1:18, 1:18]          → (N, 17, 17) 中心像素
      - 对每个方向 (dr, dc)，取出对应邻居切片:
            neighbor = imgs[:, 1+dr:18+dr, 1+dc:18+dc]
        neighbor 与 center 形状完全相同，可以逐元素比较
      - (neighbor >= center) 得到 bool 矩阵，左移 bit 位后累加到 lbp

    参数: imgs  shape (N, 19, 19)，原始像素值
    返回: lbp   shape (N, 17, 17)，uint8，每个元素是该位置的 LBP 码 0~255
    """
    N = imgs.shape[0]

    # 内部 17×17 的中心像素矩阵
    center = imgs[:, 1:18, 1:18]            # shape (N, 17, 17)

    lbp = np.zeros((N, 17, 17), dtype=np.uint8)

    for bit, (dr, dc) in enumerate(_NEIGHBOR_OFFSETS):
        # 取出该方向的邻居像素切片
        # 中心在 imgs[1:18, 1:18]，偏移 (dr,dc) 后邻居在 imgs[1+dr:18+dr, 1+dc:18+dc]
        # 例：bit0 正上方 dr=-1, dc=0  → imgs[:, 0:17, 1:18]
        #     bit2 正右   dr= 0, dc=1  → imgs[:, 1:18, 2:19]
        neighbor = imgs[:, 1 + dr : 18 + dr, 1 + dc : 18 + dc]

        # 邻居 >= 中心 → 该 bit 为 1；用位移累加到 lbp
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp    # (N, 17, 17)，值域 0~255


def extract_lbp_features(X, n_row=3, n_col=3):
    """
    从原始像素矩阵提取 LBP 空间直方图特征。

    完整流程：
      1. 把 (N, 361) 还原成 (N, 19, 19) 图像
      2. 对每张图的内部 17×17 像素计算 LBP 码 → (N, 17, 17)
      3. 用均匀 LBP 查找表把 0~255 映射到 0~58 → (N, 17, 17)
      4. 把 17×17 的 LBP 图划分成 n_row×n_col 个 cell
      5. 对每个 cell 统计 59-bin 直方图，L1 归一化（÷ cell 内像素总数）
      6. 拼接所有 cell 直方图 → (N, n_row × n_col × 59)

    L1 归一化的意义：消除 cell 大小不一致导致的数值差异，
    并使特征对图像整体亮度变化不敏感（因为 LBP 已经是相对比较，归一化后只看"各纹理占比"）。

    参数:
      X      shape (N, 361)，原始像素（缩放前调用）
      n_row  垂直方向 cell 数量（默认 3）
      n_col  水平方向 cell 数量（默认 3）

    返回:
      shape (N, n_row × n_col × 59)
      默认 3×3=9 cells，每 cell 59 bins → 531 维
    """
    N = X.shape[0]
    imgs = X.reshape(N, 19, 19)

    # ── Step A：计算 LBP 原始码 ──────────────────────────
    lbp_raw = _compute_lbp_codes(imgs)         # (N, 17, 17)，值 0~255

    # ── Step B：映射到均匀 bin ──────────────────────────
    lut = _make_uniform_lut()
    lbp_uniform = lut[lbp_raw]                 # (N, 17, 17)，值 0~58

    # ── Step C：空间直方图 ──────────────────────────────
    # 把 17×17 的 LBP 图均匀切成 n_row × n_col 个 cell
    # linspace(0, 17, n_row+1) 给出 n_row+1 个边界坐标
    row_edges = np.round(np.linspace(0, 17, n_row + 1)).astype(int)
    col_edges = np.round(np.linspace(0, 17, n_col + 1)).astype(int)

    hist_blocks = []

    for i in range(n_row):
        for j in range(n_col):
            r0, r1 = row_edges[i],   row_edges[i + 1]
            c0, c1 = col_edges[j],   col_edges[j + 1]

            # 取出该 cell 的 LBP 值，压成 (N, cell_pixels)
            cell = lbp_uniform[:, r0:r1, c0:c1].reshape(N, -1)

            # 统计每个 bin（0~58）在该 cell 内出现的次数
            # np.bincount 对单个样本有效，用 list comprehension 批量处理
            hist = np.vstack([
                np.bincount(cell[n], minlength=59) for n in range(N)
            ]).astype(np.float64)              # (N, 59)

            # L1 归一化：÷ cell 内像素总数
            # 使不同大小 cell 的直方图具有可比性
            pixel_count = (r1 - r0) * (c1 - c0)
            hist /= pixel_count                # 每个 bin 变为"频率"而非"计数"

            hist_blocks.append(hist)

    # 拼接所有 cell 的直方图：(N, n_row × n_col × 59)
    return np.hstack(hist_blocks)


# ─────────────────────── 3. 缩放 ───────────────────────

class MinMaxScaler:
    """线性缩放到 [lo, hi]；必须用训练集参数缩放测试集。"""
    def __init__(self, lo=-1.0, hi=1.0):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
        self.rng  = self.xmax - self.xmin
        self.rng[self.rng == 0] = 1.0
        return self

    def transform(self, X):
        Xs = (X - self.xmin) / self.rng
        return Xs * (self.hi - self.lo) + self.lo


# ─────────────────────── 4. 网格搜索 + 交叉验证（10-fold）───────────────────────

def cv_accuracy(y, X, C, gamma, nfold=10):
    param = f"-s 0 -t 2 -c {C} -g {gamma} -v {nfold} -q"
    with _suppress_stdout():
        return svm_train(y.tolist(), X.tolist(), param)


def grid_search(y, X,
                C_exps=range(-5, 16, 2),
                g_exps=range(-15, 4, 2),
                nfold=10,
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
            print(f"    C=2^{ce:+6.2f}={C:10.4g}  γ=2^{ge:+6.2f}={g:10.4g}  acc={acc:6.3f}%{flag}")

    print(f"[{tag}结论] 最佳 C=2^{best['ce']:+.2f}={best['C']}, "
          f"γ=2^{best['ge']:+.2f}={best['g']},  CV准确率={best['acc']:.3f}%")
    return best, grid, C_list, g_list


def fine_grid_around(best, y, X, step=0.25, span=1.0, nfold=10):
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
    # Step 1 —— 加载原始数据
    import argparse
    parser = argparse.ArgumentParser(description="Historical face/non-face SVM experiment; supply your own authorized MAT data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/face")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    Xtr, ytr, Xte = load_data(args.data_dir)

    # ── Step 1.5：提取 LBP 特征（必须在缩放前，对原始像素操作）──────────
    #
    # 为什么在缩放前提取？
    #   LBP 只比较邻居大小关系，不受像素绝对值影响。
    #   同一个严格单调灰度变换保序；逐像素坐标分别拟合的 MinMax 不保证邻域比较不变。
    #   在缩放前提取更符合"先提特征，再对特征做归一化"的规范流程。
    #
    # 每张图独立提取 LBP；下方全训练集缩放后再 CV 的历史写法仍有折内信息泄漏。
    print("[LBP] 正在提取训练集 LBP 特征...")
    Xtr_lbp = extract_lbp_features(Xtr, n_row=3, n_col=3)   # (N_train, 531)
    print("[LBP] 正在提取测试集 LBP 特征...")
    Xte_lbp = extract_lbp_features(Xte, n_row=3, n_col=3)   # (N_test,  531)
    print(f"[LBP] 特征维数: {Xtr_lbp.shape[1]}  "
          f"(3×3 cells × 59 uniform bins = {3*3*59})")
    # ─────────────────────────────────────────────────────────────────

    # Step 2 —— 对 LBP 特征做 MinMax 缩放
    # LBP 直方图各 bin 已经是频率（0~1 之间），再做缩放主要是统一量纲。
    scaler  = MinMaxScaler(-1.0, 1.0).fit(Xtr_lbp)
    Xtr_s   = scaler.transform(Xtr_lbp)
    Xte_s   = scaler.transform(Xte_lbp)
    print(f"[缩放] 训练集范围: [{Xtr_s.min():.3f}, {Xtr_s.max():.3f}]")
    print(f"[缩放] 测试集范围: [{Xte_s.min():.3f}, {Xte_s.max():.3f}]")

    # Step 3+4 —— 粗网格 + 细网格（10-fold CV）
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
    out_path = os.path.join(args.output_dir, "task3_result_LBP.txt")
    store_txt(y_pred, out_path)
    print(f"[输出] 共 {len(y_pred)} 个样本，提交格式已保存到 {out_path}")
    print(f"[输出] 预测 +1: {(y_pred == 1).sum()},  -1: {(y_pred == -1).sum()}")


if __name__ == "__main__":
    main()
