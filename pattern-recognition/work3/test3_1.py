# Public archive copy. Source mapping and any changes: ../docs/SOURCE_MANIFEST.json
"""
work3: 19x19 图像二分类 SVM
参考 SVMguide (Hsu, Chang, Lin) 的流程组织；历史代码在全训练集缩放后再 CV，存在折内统计信息泄漏:
  1) 转换数据格式
  2) 简单缩放 (scaling) 到 [-1, +1]
  3) 选择 RBF 核
  4) 交叉验证 + 网格搜索找最佳 (C, gamma)
  5) 用最佳参数在全训练集上重训
  6) 预测测试集

方法 1：扩大细网格范围

    你现在细网格只在粗网格最优点附近：

    span=1.0
    step=0.25

    可以试试：

    best, _, _, _ = fine_grid_around(
        best, ytr, Xtr_s,
        step=0.125,
        span=2.0
    )

    这样搜索更细、更宽。

    代价是运行时间变长，但通常是最直接的提分方法。

"""

import os
import sys
import io
import contextlib
import numpy as np
import scipy.io as sio
from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter


def _suppress_stdout():
    """
    屏蔽 libsvm 在 -v 模式下的 Python 级 print('Cross Validation Accuracy = X%')。
    用 Python 级 redirect_stdout 而非 os.dup2,避免某些 Windows/PowerShell
    环境下 sys.stdout 底层 Windows 句柄被 dup2 搞失效导致 [WinError 6]。
    libsvm 的 C 层输出已由 -q 压制,Python 级重定向就够用。
    """
    return contextlib.redirect_stdout(io.StringIO())

# 数据与脚本同目录(脚本就在 work3/ 下)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────── 1. 数据加载 ───────────────────────

def _load_mat(path):
    """加载 .mat 文件,自动取第一个非私有变量"""
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith("_")][0]
    return np.asarray(mat[key])


def load_data(data_dir=DATA_DIR):
    """
    读取:
      train_data  : (N_train, 361) 或 (361, N_train) —— 19x19 图像按列/行堆叠
      train_label : (N_train,)     取值 +1 / -1
      test_data   : (N_test, 361)  或 (361, N_test)
    ReadMe 原文说 "每一列代表一个样本",因此若列数 = 361 则已经是 (N, 361),
    否则需要转置。
    """
    files = {
        "train_data":  os.path.join(data_dir, "train_data.mat"),
        "train_label": os.path.join(data_dir, "train_label.mat"),
        "test_data":   os.path.join(data_dir, "test_data.mat"),
    }
    for name, p in files.items():
        if not os.path.isfile(p):
            sys.exit(f"[错误] 找不到 {p},请把 {name}.mat 放到 work3/ 下")

    Xtr = _load_mat(files["train_data"]).astype(np.float64)
    ytr = _load_mat(files["train_label"]).astype(np.float64).ravel()
    Xte = _load_mat(files["test_data"]).astype(np.float64)

    # ReadMe: 每列一个样本 => 需转置成 (N, D)
    if Xtr.shape[0] == 361 and Xtr.shape[1] != 361:
        Xtr = Xtr.T
    if Xte.shape[0] == 361 and Xte.shape[1] != 361:
        Xte = Xte.T

    print(f"[加载] Xtrain={Xtr.shape}, ytrain={ytr.shape}, Xtest={Xte.shape}")
    print(f"[加载] 训练集 +1: {(ytr == 1).sum()},  -1: {(ytr == -1).sum()}")
    return Xtr, ytr, Xte


# ─────────────────────── 2. 缩放 (Section 2.2) ───────────────────────

class MinMaxScaler:
    """
    线性缩放到 [lo, hi]。SVMguide 特别强调:必须用训练集的缩放参数去缩放测试集。
    """
    def __init__(self, lo=-1.0, hi=1.0):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
        self.rng = self.xmax - self.xmin
        self.rng[self.rng == 0] = 1.0    # 常数特征保持不动
        return self

    def transform(self, X):
        Xs = (X - self.xmin) / self.rng       # -> [0, 1]
        return Xs * (self.hi - self.lo) + self.lo


# ─────────────────────── 3+4. 网格搜索 + 交叉验证 (Section 3.2) ───────────────────────

def cv_accuracy(y, X, C, gamma, nfold=5):
    """libsvm 的 -v 交叉验证,返回 accuracy (%)。"""
    param = f"-s 0 -t 2 -c {C} -g {gamma} -v {nfold} -q"
    with _suppress_stdout():
        return svm_train(y.tolist(), X.tolist(), param)


def grid_search(y, X,
                C_exps=range(-5, 16, 2),       # 2^-5, 2^-3, ..., 2^15
                g_exps=range(-15, 4, 2),       # 2^-15, 2^-13, ..., 2^3
                nfold=5,
                tag="粗网格"):
    """
    指数增长序列网格搜索(严格按指南推荐)。
    返回: (best_C, best_gamma, best_acc, grid 结果矩阵)
    """
    print(f"\n=== {tag}网格搜索 ===")
    print(f"    C 取值  : 2^{{{list(C_exps)}}}")
    print(f"    γ 取值  : 2^{{{list(g_exps)}}}")
    print(f"    交叉验证: {nfold}-fold")

    best = {"acc": -1, "C": None, "g": None, "ce": None, "ge": None}
    grid = np.full((len(list(C_exps)), len(list(g_exps))), np.nan)
    C_list = list(C_exps); g_list = list(g_exps)

    for i, ce in enumerate(C_list):
        for j, ge in enumerate(g_list):
            C, g = 2.0 ** ce, 2.0 ** ge
            acc = cv_accuracy(y, X, C, g, nfold)
            grid[i, j] = acc
            flag = ""
            if acc > best["acc"]:
                best.update(acc=acc, C=C, g=g, ce=ce, ge=ge)
                flag = "  <-- best"
            print(f"    C=2^{ce:+6.2f}={C:10.4g}  γ=2^{ge:+6.2f}={g:10.4g}  acc={acc:6.3f}%{flag}")

    print(f"[{tag}结论] 最佳 C=2^{best['ce']:+.2f}={best['C']}, "
          f"γ=2^{best['ge']:+.2f}={best['g']},  CV准确率={best['acc']:.3f}%")
    return best, grid, C_list, g_list


def fine_grid_around(best, y, X, step=0.125, span=2.0, nfold=5):
    """
    在粗网格最优点附近做 0.25 步长的细网格搜索(SVMguide Fig.3 的做法)。
    """
    ce, ge = best["ce"], best["ge"]
    C_exps = np.arange(ce - span, ce + span + 1e-9, step)
    g_exps = np.arange(ge - span, ge + span + 1e-9, step) #ce + span + 1e-9 是为了确保右边界被包含（因为浮点误差）
    return grid_search(y, X, C_exps=C_exps, g_exps=g_exps,
                       nfold=nfold, tag="细")


# ─────────────────────── 5+6. 训练 & 预测 ───────────────────────

def train_final(y, X, C, gamma):
    """用最佳超参在全训练集上训练最终模型。"""
    prob = svm_problem(y.tolist(), X.tolist())
    param = svm_parameter(f"-s 0 -t 2 -c {C} -g {gamma} -q")
    with _suppress_stdout():
        return svm_train(prob, param)


def predict(model, X, y_dummy=None):
    """对测试集预测。若没有测试标签,用零占位。"""
    if y_dummy is None:
        y_dummy = [0.0] * X.shape[0]
    p_lab, p_acc, p_val = svm_predict(y_dummy, X.tolist(), model, "-q")
    return np.asarray(p_lab)


def store_txt(labels, result_txt):
    """按提交格式写文件: 每行 `imageID label`,空格分隔,imageID 从 1 开始。"""
    with open(result_txt, "w") as w:
        for index, lab in enumerate(labels, 1):
            w.write(f"{index} {int(lab)}\n")


# ─────────────────────── 主流程 ───────────────────────

def main():
    # step 1 —— 加载 & 转换格式
    import argparse
    parser = argparse.ArgumentParser(description="Historical face/non-face SVM experiment; supply your own authorized MAT data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/face")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    Xtr, ytr, Xte = load_data(args.data_dir)

    # step 2 —— 缩放(同一参数应用到测试集)
    scaler = MinMaxScaler(-1.0, 1.0).fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    print(f"[缩放] 训练集范围: [{Xtr_s.min():.3f}, {Xtr_s.max():.3f}]")
    print(f"[缩放] 测试集范围: [{Xte_s.min():.3f}, {Xte_s.max():.3f}]  "
          f"(可超出 [-1,1] 属正常)")

    # step 3+4 —— RBF 核 + 粗网格 + 细网格
    best, _, _, _ = grid_search(ytr, Xtr_s)
    best, _, _, _ = fine_grid_around(best, ytr, Xtr_s)

    # step 5 —— 全训练集重训
    print(f"\n=== 用最佳参数在全训练集上训练 ===")
    print(f"    C={best['C']},  γ={best['g']}")
    model = train_final(ytr, Xtr_s, best["C"], best["g"])

    # 训练集精度(仅作参考,不代表泛化能力)
    p_train = predict(model, Xtr_s, ytr.tolist())
    print(f"[训练集精度] {(p_train == ytr).mean() * 100:.3f}%")

    # step 6 —— 预测测试集,按提交格式 `imageID label` 写文件
    y_pred = predict(model, Xte_s)
    out_path = os.path.join(args.output_dir, "task3_result_1.txt")
    store_txt(y_pred, out_path)
    print(f"[输出] 共 {len(y_pred)} 个样本,提交格式已保存到 {out_path}")
    print(f"[输出] 预测 +1: {(y_pred == 1).sum()},  -1: {(y_pred == -1).sum()}")


if __name__ == "__main__":
    main()
