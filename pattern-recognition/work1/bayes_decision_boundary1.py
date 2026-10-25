"""
二维正态分布贝叶斯决策面可视化
- 线性决策面：两类协方差矩阵相同（Σ1 = Σ2）
- 非线性决策面：两类协方差矩阵不同（Σ1 ≠ Σ2）

判别函数（对数后验概率）：
  gi(x) = -1/2 * (x-μi)^T Σi^{-1} (x-μi) - 1/2 * log|Σi| + log P(ωi)

决策面：g1(x) = g2(x)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def log_discriminant(X, mu, sigma, prior):
    """对数判别函数 gi(x)"""
    inv_sigma = np.linalg.inv(sigma)
    sign, log_det = np.linalg.slogdet(sigma)
    diff = X - mu                                          # (N, 2)
    mahal = np.einsum('ni,ij,nj->n', diff, inv_sigma, diff)  # 马氏距离
    return -0.5 * mahal - 0.5 * log_det + np.log(prior)


def make_grid(means, padding=3, resolution=300):
    """根据均值自动生成网格范围"""
    all_pts = np.vstack(means)
    xmin, ymin = all_pts.min(axis=0) - padding
    xmax, ymax = all_pts.max(axis=0) + padding
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, resolution),
        np.linspace(ymin, ymax, resolution)
    )
    return xx, yy


def draw_ellipse(ax, mu, sigma, color, n_std=(1, 2)):
    """绘制协方差椭圆（1σ、2σ）"""
    vals, vecs = np.linalg.eigh(sigma)
    angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))
    for ns in n_std:
        width, height = 2 * ns * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=width, height=height,
                      angle=angle, edgecolor=color,
                      facecolor='none', linewidth=1.5,
                      linestyle='--', alpha=0.7)
        ax.add_patch(ell)


def plot_scene(ax, mu1, mu2, sigma1, sigma2, prior1, prior2,
               title, n_samples=300):
    """通用绘图函数"""
    prior2_ = 1 - prior1 if prior2 is None else prior2

    # ── 采样点 ──
    X1 = multivariate_normal(mu1, sigma1).rvs(n_samples)
    X2 = multivariate_normal(mu2, sigma2).rvs(n_samples)

    # ── 网格 ──
    xx, yy = make_grid([mu1, mu2], padding=4)
    grid = np.c_[xx.ravel(), yy.ravel()]

    g1 = log_discriminant(grid, mu1, sigma1, prior1).reshape(xx.shape)
    g2 = log_discriminant(grid, mu2, sigma2, prior2_).reshape(xx.shape)
    diff = g1 - g2                         # >0 → ω1，<0 → ω2

    # ── 背景着色 ──
    ax.contourf(xx, yy, diff, levels=[-1e9, 0, 1e9],
                colors=['#AED6F1', '#ABEBC6'], alpha=0.35)

    # ── 决策面 ──
    ax.contour(xx, yy, diff, levels=[0],
               colors='red', linewidths=2.5)

    # ── 等概率密度轮廓 ──
    for mu, sigma, c in [(mu1, sigma1, '#1A5276'), (mu2, sigma2, '#145A32')]:
        pdf = multivariate_normal(mu, sigma).pdf(
            np.dstack([xx, yy]))
        ax.contour(xx, yy, pdf, levels=5,
                   colors=[c], alpha=0.4, linewidths=0.8)
        draw_ellipse(ax, mu, sigma, c)

    # ── 散点 ──
    ax.scatter(X1[:, 0], X1[:, 1], s=12, alpha=0.5,
               color='#2980B9', label=f'ω₁  P={prior1:.2f}')
    ax.scatter(X2[:, 0], X2[:, 1], s=12, alpha=0.5,
               color='#27AE60', label=f'ω₂  P={prior2_:.2f}')

    # ── 均值标记 ──
    ax.plot(*mu1, 'b*', markersize=14, zorder=5)
    ax.plot(*mu2, 'g*', markersize=14, zorder=5)

    # 添加虚拟图例条目表示决策面
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2.5, label='决策面'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + ['决策面'],
              fontsize=9, loc='upper right')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)


# ──────────────────────────────────────────────
# 场景定义
# ──────────────────────────────────────────────

def case_linear_equal_prior():
    """线性 - 等先验：Σ1=Σ2，P(ω1)=P(ω2)=0.5"""
    mu1 = np.array([-2.0,  0.0])
    mu2 = np.array([ 2.0,  0.0])
    sigma = np.array([[1.5, 0.5], [0.5, 1.0]])
    return mu1, mu2, sigma, sigma, 0.5, 0.5


def case_linear_unequal_prior():
    """线性 - 不等先验：Σ1=Σ2，P(ω1)≠P(ω2)，决策面偏移"""
    mu1 = np.array([-2.0,  0.0])
    mu2 = np.array([ 2.0,  0.0])
    sigma = np.array([[1.5, 0.5], [0.5, 1.0]])
    return mu1, mu2, sigma, sigma, 0.3, 0.7


def case_nonlinear_diff_variance():
    """非线性 - 不同方差：Σ2 远大于 Σ1，产生闭合曲线"""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([0.0, 0.0])          # 同均值，协方差不同
    sigma1 = np.array([[0.5, 0.0], [0.0, 0.5]])
    sigma2 = np.array([[3.0, 0.0], [0.0, 3.0]])
    return mu1, mu2, sigma1, sigma2, 0.5, 0.5


def case_nonlinear_rotated():
    """非线性 - 旋转协方差：产生双曲线/椭圆形决策面"""
    mu1 = np.array([-1.5,  1.5])
    mu2 = np.array([ 1.5, -1.5])
    sigma1 = np.array([[2.0,  1.5], [ 1.5, 2.0]])
    sigma2 = np.array([[2.0, -1.5], [-1.5, 2.0]])
    return mu1, mu2, sigma1, sigma2, 0.5, 0.5


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def plot_linear():
    """绘制线性决策面图像"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('线性决策面 - 贝叶斯分类（Σ₁=Σ₂）',
                 fontsize=16, fontweight='bold', y=0.98)

    scenes = [
        (axes[0], *case_linear_equal_prior(),
         '线性决策面\nΣ₁=Σ₂，P(ω₁)=P(ω₂)=0.5'),

        (axes[1], *case_linear_unequal_prior(),
         '线性决策面（偏移）\nΣ₁=Σ₂，P(ω₁)=0.3，P(ω₂)=0.7'),
    ]

    np.random.seed(42)
    for ax, mu1, mu2, s1, s2, p1, p2, title in scenes:
        plot_scene(ax, mu1, mu2, s1, s2, p1, p2, title)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('bayes_linear_boundary.png', dpi=150, bbox_inches='tight')
    print("线性决策面图像已保存到 bayes_linear_boundary.png")
    plt.show()


def plot_nonlinear():
    """绘制非线性决策面图像"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('非线性决策面 - 贝叶斯分类（Σ₁≠Σ₂）',
                 fontsize=16, fontweight='bold', y=0.98)

    scenes = [
        (axes[0], *case_nonlinear_diff_variance(),
         '非线性决策面（闭合曲线）\nΣ₁≠Σ₂，均值相同，方差差异大'),

        (axes[1], *case_nonlinear_rotated(),
         '非线性决策面（双曲型）\nΣ₁≠Σ₂，旋转协方差'),
    ]

    np.random.seed(42)
    for ax, mu1, mu2, s1, s2, p1, p2, title in scenes:
        plot_scene(ax, mu1, mu2, s1, s2, p1, p2, title)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('bayes_nonlinear_boundary.png', dpi=150, bbox_inches='tight')
    print("非线性决策面图像已保存到 bayes_nonlinear_boundary.png")
    plt.show()


def main():
    plot_linear()
    plot_nonlinear()


if __name__ == '__main__':
    main()
