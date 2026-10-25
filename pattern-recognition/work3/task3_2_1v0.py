import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 工具函数 1：打乱数据
# =========================================================
def shuffle_data(X, y, seed=0):
    """
    X: 特征矩阵，形状为 [样本数, 2]
    y: 标签数组，形状为 [样本数]
    """
    rng = np.random.default_rng(seed)

    index = np.arange(len(y))
    rng.shuffle(index)

    X = X[index]
    y = y[index]

    return X, y


# =========================================================
# 工具函数 2：保存成 LIBSVM 格式
# =========================================================
def save_libsvm(X, y, file_path):
    """
    LIBSVM 格式：
    label 1:x1 2:x2

    例如：
    +1 1:2.35 2:1.28
    -1 1:-1.23 2:-2.45
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(len(y)):
            label = int(y[i])
            x1 = X[i, 0]
            x2 = X[i, 1]

            line = f"{label} 1:{x1:.6f} 2:{x2:.6f}\n"
            f.write(line)


# =========================================================
# 工具函数 3：保存成 NumPy .npz 格式
# =========================================================
def save_numpy(X, y, file_path):
    """
    保存为 .npz 文件，加载方式：
        data = np.load("xxx.npz")
        X, y = data["X"], data["y"]
    """
    np.savez(file_path, X=X, y=y)


# =========================================================
# 工具函数 4：画二维散点图
# =========================================================
def plot_data(X, y, title="2D Data", save_path=None):
    """
    画出二维样本点；若指定 save_path 则同时保存为图片文件。
    """
    plt.figure(figsize=(6, 6))

    # 正类
    pos = (y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker="o", label="+1")

    # 负类
    neg = (y == -1)
    plt.scatter(X[neg, 0], X[neg, 1], marker="x", label="-1")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =========================================================
# 类型 1：线性可分数据
# =========================================================
def make_linear_separable(n_per_class=50, seed=0):
    """
    生成线性可分二维数据。

    正类中心在 (2, 2)
    负类中心在 (-2, -2)

    spread 越小，数据越集中，越容易线性可分。
    """
    rng = np.random.default_rng(seed)

    # 每一类的分散程度
    spread = 0.6

    # 正类 +1
    X_pos = rng.normal(
        loc=[2, 2],
        scale=spread,
        size=(n_per_class, 2)
    )

    # 负类 -1
    X_neg = rng.normal(
        loc=[-2, -2],
        scale=spread,
        size=(n_per_class, 2)
    )

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(n_per_class),
        -np.ones(n_per_class)
    ])

    X, y = shuffle_data(X, y, seed)

    return X, y


# =========================================================
# 类型 2：重叠型不可分数据
# =========================================================
def make_overlap(n_per_class=50, seed=1):
    """
    生成两类相互重叠的数据。

    正类中心在 (0.8, 0.8)
    负类中心在 (-0.8, -0.8)

    spread 较大，所以两类会产生重叠。
    """
    rng = np.random.default_rng(seed)

    spread = 1.2

    X_pos = rng.normal(
        loc=[0.8, 0.8],
        scale=spread,
        size=(n_per_class, 2)
    )

    X_neg = rng.normal(
        loc=[-0.8, -0.8],
        scale=spread,
        size=(n_per_class, 2)
    )

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(n_per_class),
        -np.ones(n_per_class)
    ])

    X, y = shuffle_data(X, y, seed)

    return X, y


# =========================================================
# 类型 3：圆环型不可分数据
# =========================================================
def make_circle(n_per_class=50, seed=2):
    """
    生成圆环型数据。

    负类 -1：内圈
    正类 +1：外圈

    这种数据无法用直线分开，适合用 RBF 核。
    """
    rng = np.random.default_rng(seed)

    # -------------------------
    # 内圈：负类 -1
    # -------------------------
    theta_inner = rng.uniform(0, 2 * np.pi, n_per_class)
    r_inner = rng.normal(loc=0.8, scale=0.15, size=n_per_class)

    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)

    X_neg = np.column_stack([x_inner, y_inner])

    # -------------------------
    # 外圈：正类 +1
    # -------------------------
    theta_outer = rng.uniform(0, 2 * np.pi, n_per_class)
    r_outer = rng.normal(loc=2.0, scale=0.15, size=n_per_class)

    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)

    X_pos = np.column_stack([x_outer, y_outer])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(n_per_class),
        -np.ones(n_per_class)
    ])

    X, y = shuffle_data(X, y, seed)

    return X, y


# =========================================================
# 类型 4：XOR 型不可分数据
# =========================================================
def make_xor(n_per_class=50, seed=3):
    """
    生成 XOR 数据。

    正类 +1：
        右上角
        左下角

    负类 -1：
        左上角
        右下角

    这种数据也不能被一条直线分开。
    """
    rng = np.random.default_rng(seed)

    spread = 0.35
    n_half = n_per_class // 2

    # 正类：右上角和左下角
    X_pos_1 = rng.normal(loc=[1.5, 1.5], scale=spread, size=(n_half, 2))
    X_pos_2 = rng.normal(loc=[-1.5, -1.5], scale=spread, size=(n_per_class - n_half, 2))

    X_pos = np.vstack([X_pos_1, X_pos_2])

    # 负类：左上角和右下角
    X_neg_1 = rng.normal(loc=[-1.5, 1.5], scale=spread, size=(n_half, 2))
    X_neg_2 = rng.normal(loc=[1.5, -1.5], scale=spread, size=(n_per_class - n_half, 2))

    X_neg = np.vstack([X_neg_1, X_neg_2])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(n_per_class),
        -np.ones(n_per_class)
    ])

    X, y = shuffle_data(X, y, seed)

    return X, y


# =========================================================
# 类型 5：月牙型数据
# =========================================================
def make_moons(n_per_class=50, noise=0.1, seed=4):
    """
    生成两个交错半圆（月牙型）数据。

    正类 +1：上半圆（圆心在原点，半径 1）
    负类 -1：下半圆（圆心偏移 (1, -0.5)，半径 1）

    两类交错但不重叠（noise 小时），线性核和多项式核都难以分开，
    RBF 核能轻松处理。用于对比圆环/XOR 强调 RBF 核的泛化优势。

    noise：高斯噪声标准差，越大两类越混叠
    """
    rng = np.random.default_rng(seed)

    # 正类：上半圆，角度 [0, π]
    theta_pos = rng.uniform(0, np.pi, n_per_class)
    X_pos = np.column_stack([
        np.cos(theta_pos),
        np.sin(theta_pos)
    ])

    # 负类：下半圆，圆心偏移 (1, -0.5) 使两类交错
    theta_neg = rng.uniform(0, np.pi, n_per_class)
    X_neg = np.column_stack([
        1.0 - np.cos(theta_neg),
        -0.5 - np.sin(theta_neg)   # 向下翻转并偏移
    ])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(n_per_class),
        -np.ones(n_per_class)
    ])

    # 加高斯噪声，模拟真实数据的不确定性
    X += rng.normal(0, noise, X.shape)

    X, y = shuffle_data(X, y, seed)
    return X, y


# =========================================================
# 主程序：生成并保存数据
# =========================================================
if __name__ == "__main__":

    # 每一类 50 个样本，总共 100 个样本
    # 题目要求总样本数不少于 60，所以这里是满足要求的
    n_per_class = 50

    # 1. 线性可分数据
    X1, y1 = make_linear_separable(n_per_class=n_per_class, seed=0)
    save_libsvm(X1, y1, "linear_separable.txt")
    save_numpy(X1, y1, "linear_separable.npz")
    plot_data(X1, y1, "Linear Separable Data", save_path="linear_separable.png")

    # 2. 重叠型不可分数据
    X2, y2 = make_overlap(n_per_class=n_per_class, seed=1)
    save_libsvm(X2, y2, "overlap.txt")
    save_numpy(X2, y2, "overlap.npz")
    plot_data(X2, y2, "Overlap Nonlinear / Nonseparable Data", save_path="overlap.png")

    # 3. 圆环型不可分数据
    X3, y3 = make_circle(n_per_class=n_per_class, seed=2)
    save_libsvm(X3, y3, "circle.txt")
    save_numpy(X3, y3, "circle.npz")
    plot_data(X3, y3, "Circle Nonlinear Data", save_path="circle.png")

    # 4. XOR 型不可分数据
    X4, y4 = make_xor(n_per_class=n_per_class, seed=3)
    save_libsvm(X4, y4, "xor.txt")
    save_numpy(X4, y4, "xor.npz")
    plot_data(X4, y4, "XOR Nonlinear Data", save_path="xor.png")

    # 5. 月牙型数据（线性/多项式核难以分开，RBF 核专属）
    X5, y5 = make_moons(n_per_class=n_per_class, noise=0.5, seed=4)
    save_libsvm(X5, y5, "moons.txt")
    save_numpy(X5, y5, "moons.npz")
    plot_data(X5, y5, "Moons Data (RBF Kernel Favored)", save_path="moons.png")