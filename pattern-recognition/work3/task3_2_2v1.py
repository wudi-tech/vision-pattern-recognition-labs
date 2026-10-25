import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_train, svm_predict, svm_read_problem
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 按系统可用的选
plt.rcParams['axes.unicode_minus'] = False   # 防止负号显示异常

# =========================================================
# 1. 把 LIBSVM 读取出来的字典特征转成 numpy 数组
# =========================================================
def libsvm_x_to_numpy(x):
    """
    LIBSVM 读取出的 x 是这样的：
    [
        {1: 2.1, 2: 3.2},
        {1: -1.5, 2: -2.3},
        ...
    ]

    这里把它转成二维 numpy 数组：
    [
        [2.1, 3.2],
        [-1.5, -2.3],
        ...
    ]
    """
    X = np.zeros((len(x), 2))

    for i, sample in enumerate(x):
        X[i, 0] = sample.get(1, 0)
        X[i, 1] = sample.get(2, 0)

    return X


# =========================================================
# 2. 把 numpy 数组转成 LIBSVM 需要的字典格式
# =========================================================
def numpy_to_libsvm_x(X):
    """
    把 numpy 格式：
    [
        [x1, x2],
        [x1, x2]
    ]

    转成 LIBSVM 格式：
    [
        {1: x1, 2: x2},
        {1: x1, 2: x2}
    ]
    """
    x_list = []

    for i in range(X.shape[0]):
        x_list.append({
            1: float(X[i, 0]),
            2: float(X[i, 1])
        })

    return x_list


# =========================================================
# 3. 取出支持向量
# =========================================================
def get_support_vectors(model):
    """
    从 LIBSVM 模型中取出支持向量，并转成 numpy 数组。
    """
    sv_dict_list = model.get_SV()
    SV = np.zeros((len(sv_dict_list), 2))

    for i, sample in enumerate(sv_dict_list):
        SV[i, 0] = sample.get(1, 0)
        SV[i, 1] = sample.get(2, 0)

    return SV


# =========================================================
# 3.5 MinMaxScaler（不依赖 sklearn，仅用 numpy）
# =========================================================
class _MinMaxScaler:
    """把特征线性缩放到 [lo, hi]，必须用训练集参数缩放测试集/网格。"""
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

    def inverse_transform(self, Xs):
        """将缩放后的坐标还原到原始空间（用于把支持向量映射回原始坐标绘图）。"""
        return (Xs - self.lo) / (self.hi - self.lo) * self.rng + self.xmin


# =========================================================
# 4. 训练并可视化
# =========================================================
def train_and_plot(file_name, option_str, title_name, grid_num=250, scale=False):
    """
    scale=False（默认）：直接用原始数据训练，与之前行为一致。
    scale=True ：训练前对特征做 MinMax 缩放到 [-1, 1]，
                 网格点也用同一 scaler 缩放后再预测，
                 支持向量逆变换回原始坐标再绘图。
    """
    # 1. 读取数据
    y, x = svm_read_problem(file_name)
    X    = libsvm_x_to_numpy(x)
    y_np = np.array(y)

    # 缩放（scale=True）：对训练特征做 MinMax 缩放到 [-1, 1]
    # 同一 scaler 之后用于处理网格点（预测）和支持向量（绘图逆变换）
    if scale:
        scaler  = _MinMaxScaler(-1.0, 1.0).fit(X)
        X_fit   = scaler.transform(X)
        title_name = title_name + "  [已缩放]"
    else:
        scaler = None
        X_fit  = X

    x_fit = numpy_to_libsvm_x(X_fit)

    # 2. 训练模型（使用缩放后或原始特征）
    model = svm_train(y, x_fit, option_str)

    # 3. 训练集预测
    pred_label, acc, _ = svm_predict(y, x_fit, model, "-q")

    print("=" * 60)
    print("数据集：", file_name)
    print("参数：  ", option_str)
    print("缩放：  ", "是 [-1, 1]" if scale else "否")
    print("训练准确率：%.2f%%" % acc[0])
    print("支持向量数量：", model.get_nr_sv())
    print("=" * 60)

    # 4. 构造网格点（坐标系始终在原始特征空间，用于绘图）
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, grid_num),
        np.linspace(x2_min, x2_max, grid_num)
    )
    grid_X = np.c_[xx1.ravel(), xx2.ravel()]

    # 若已缩放，对网格点做相同变换后送入模型预测
    grid_X_pred   = scaler.transform(grid_X) if scale else grid_X
    grid_x_libsvm = numpy_to_libsvm_x(grid_X_pred)
    dummy_y        = [0] * len(grid_x_libsvm)

    # 5. 网格点预测
    grid_pred, _, grid_decision = svm_predict(dummy_y, grid_x_libsvm, model, "-q")
    Z_pred     = np.array(grid_pred).reshape(xx1.shape)
    Z_decision = np.array(grid_decision).reshape(xx1.shape)

    # 6. 取出支持向量；若已缩放，逆变换回原始坐标用于绘图
    SV      = get_support_vectors(model)
    SV_plot = scaler.inverse_transform(SV) if scale else SV

    # 7. 绘图（坐标轴始终是原始特征空间）
    plt.figure(figsize=(7, 6))
    plt.contourf(xx1, xx2, Z_pred, alpha=0.25)
    plt.contour(xx1, xx2, Z_decision, levels=[0], linewidths=2)
    plt.contour(xx1, xx2, Z_decision, levels=[-1, 1], linestyles="--", linewidths=1)

    pos_index = y_np == 1
    plt.scatter(X[pos_index, 0], X[pos_index, 1], marker="o", s=45, label="+1 samples")

    neg_index = y_np == -1
    plt.scatter(X[neg_index, 0], X[neg_index, 1], marker="x", s=45, label="-1 samples")

    plt.scatter(
        SV_plot[:, 0], SV_plot[:, 1],
        s=120, facecolors="none", edgecolors="black",
        linewidths=1.5, label="Support Vectors"
    )

    plt.title(title_name)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# =========================================================
# 5. 主程序：分别训练不同类型的数据
# =========================================================
if __name__ == "__main__":

    # 1. 线性可分数据：线性核
    train_and_plot(
        file_name="linear_separable.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Linear Separable Data: Linear SVM"
    )

    # 2. 重叠型不可分数据：线性核，C=1
    train_and_plot(
        file_name="overlap.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Overlap Data: Linear Soft-margin SVM, C=1"
    )

    # 3. 重叠型不可分数据：C=0.1
    train_and_plot(
        file_name="overlap.txt",
        option_str="-s 0 -t 0 -c 0.1 -q",
        title_name="Overlap Data: Linear SVM, C=0.1"
    )

    # 4. 重叠型不可分数据：C=100
    train_and_plot(
        file_name="overlap.txt",
        option_str="-s 0 -t 0 -c 100 -q",
        title_name="Overlap Data: Linear SVM, C=100"
    )

    # 5. 圆环数据：线性核
    train_and_plot(
        file_name="circle.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Circle Data: Linear SVM"
    )

    # 6. 圆环数据：RBF 核
    train_and_plot(
        file_name="circle.txt",
        option_str="-s 0 -t 2 -c 10 -g 1 -q",
        title_name="Circle Data: RBF SVM"
    )

    # 7. XOR 数据：RBF 核
    train_and_plot(
        file_name="xor.txt",
        option_str="-s 0 -t 2 -c 10 -g 1 -q",
        title_name="XOR Data: RBF SVM"
    )

    # 8. XOR 数据：二次多项式核
    train_and_plot(
        file_name="xor.txt",
        option_str="-s 0 -t 1 -d 2 -c 10 -q",
        title_name="XOR Data: Polynomial Kernel SVM"
    )

    # ── 改进：缩放 vs 未缩放对比实验 ──────────────────────────────
    # 重叠数据特征量纲一致，缩放前后差异小，适合作为基准对比；
    # 圆环数据的 RBF 核对 gamma 的选取更依赖缩放，差异更明显。

    # 9. 重叠数据：未缩放 vs 已缩放，C=1
    train_and_plot(
        file_name="overlap.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Overlap: Linear SVM, C=1, No Scaling",
        scale=False
    )
    train_and_plot(
        file_name="overlap.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Overlap: Linear SVM, C=1, With Scaling",
        scale=True
    )

    # 10. 圆环数据：未缩放 vs 已缩放，RBF 核（相同 gamma）
    #     缩放后 gamma 的物理含义不同，决策边界会有明显变化
    train_and_plot(
        file_name="circle.txt",
        option_str="-s 0 -t 2 -c 10 -g 1 -q",
        title_name="Circle: RBF SVM, C=10 g=1, No Scaling",
        scale=False
    )
    train_and_plot(
        file_name="circle.txt",
        option_str="-s 0 -t 2 -c 10 -g 1 -q",
        title_name="Circle: RBF SVM, C=10 g=1, With Scaling",
        scale=True
    )

    # 11. 月牙型数据：线性核 vs RBF 核（需先运行 task3_2_1v0.py 生成 moons.txt）
    train_and_plot(
        file_name="moons.txt",
        option_str="-s 0 -t 0 -c 1 -q",
        title_name="Moons: Linear SVM (expected to fail)",
        scale=True
    )
    train_and_plot(
        file_name="moons.txt",
        option_str="-s 0 -t 2 -c 10 -g 1 -q",
        title_name="Moons: RBF SVM (should work well)",
        scale=True
    )