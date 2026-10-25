# Public archive copy. Source mapping and any changes: ../docs/SOURCE_MANIFEST.json
"""
基于贝叶斯决策的 Nemo 鱼像素点分类
任务1: 对 309.bmp 进行灰度和彩色分割
任务2: 对另外4张彩色图像进行分割
方法: 参数估计(高斯)、非参数估计(Parzen窗/KDE、k-NN)
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal
from scipy.special import logsumexp as scipy_logsumexp
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

# ─────────────────────── 数据加载 ───────────────────────

def load_data():
    """加载训练样本和掩码"""
    mat = sio.loadmat(os.path.join(DATA_DIR, "array_sample.mat"))
    # 找到包含样本的变量（非私有键）
    key = [k for k in mat if not k.startswith('_')][0]
    data = mat[key].astype(np.float64)   # (N, 5): gray, R, G, B, label

    mask_mat = sio.loadmat(os.path.join(DATA_DIR, "Mask.mat"))
    mkey = [k for k in mask_mat if not k.startswith('_')][0]
    mask = mask_mat[mkey].astype(bool)   # 2D binary mask

    labels = data[:, -1]                 # 1 or -1
    features = data[:, :-1]              # gray, R, G, B

    idx1 = labels == 1
    idx2 = labels == -1
    print(f"类别1 (鱼体): {idx1.sum()} 个样本")
    print(f"类别2 (背景): {idx2.sum()} 个样本")
    return features, labels, idx1, idx2, mask


# ─────────────────────── 参数法：高斯模型 ───────────────────────

class GaussianBayes:
    """多元高斯最大后验分类器"""

    def fit(self, features, idx1, idx2):
        n1, n2 = idx1.sum(), idx2.sum()
        n = n1 + n2
        self.prior1 = n1 / n
        self.prior2 = n2 / n

        X1, X2 = features[idx1], features[idx2]
        self.mu1 = X1.mean(axis=0)
        self.mu2 = X2.mean(axis=0)
        # 加正则化防止奇异
        reg = 1e-6 * np.eye(X1.shape[1])
        self.cov1 = np.cov(X1.T) + reg
        self.cov2 = np.cov(X2.T) + reg
        return self

    def predict(self, X):
        """返回预测标签 (1 or -1) 和后验概率差"""
        log_p1 = multivariate_normal.logpdf(X, self.mu1, self.cov1) + np.log(self.prior1)
        log_p2 = multivariate_normal.logpdf(X, self.mu2, self.cov2) + np.log(self.prior2)
        pred = np.where(log_p1 >= log_p2, 1, -1)
        return pred


class GaussianBayes1D:
    """一维高斯分类器（用于灰度图）"""

    def fit(self, features, idx1, idx2):
        n1, n2 = idx1.sum(), idx2.sum()
        self.prior1 = n1 / (n1 + n2)
        self.prior2 = n2 / (n1 + n2)

        g1 = features[idx1, 0]   # 灰度列
        g2 = features[idx2, 0]
        self.mu1, self.std1 = g1.mean(), g1.std() + 1e-6
        self.mu2, self.std2 = g2.mean(), g2.std() + 1e-6
        return self

    def predict(self, gray_vals):
        from scipy.stats import norm
        g = np.asarray(gray_vals).ravel()
        log_p1 = norm.logpdf(g, self.mu1, self.std1) + np.log(self.prior1)
        log_p2 = norm.logpdf(g, self.mu2, self.std2) + np.log(self.prior2)
        return np.where(log_p1 >= log_p2, 1, -1)


# ─────────────────────── 非参数法1：Parzen窗/KDE ───────────────────────

class ParzenBayes:
    """
    Parzen 窗贝叶斯分类器 — 朴素贝叶斯 + 逐维1D高斯KDE
    假设各特征独立：log p(x|c) = sum_j log p_j(x_j|c)
    每维用 Silverman 1D 带宽：h_j = 1.06 * std_j * n^(-1/5)
    """

    def fit(self, features, idx1, idx2):
        n1, n2 = idx1.sum(), idx2.sum()
        self.prior1 = n1 / (n1 + n2)
        self.prior2 = n2 / (n1 + n2)

        self.X1 = features[idx1].copy()   # (n1, d)
        self.X2 = features[idx2].copy()   # (n2, d)

        # 每维 Silverman 带宽
        self.h1 = 1.06 * self.X1.std(axis=0) * (n1 ** (-0.2)) + 1e-6
        self.h2 = 1.06 * self.X2.std(axis=0) * (n2 ** (-0.2)) + 1e-6
        return self

    @staticmethod
    def _log_kde_1d(xq, xtrain, h):
        """1D 高斯 KDE：xq (M,), xtrain (n,), h scalar → log p (M,)"""
        n = len(xtrain)
        # xq[:,None] - xtrain[None,:] → (M, n)
        diff = xq[:, None] - xtrain[None, :]      # (M, n)
        log_w = -0.5 * (diff / h) ** 2 - np.log(h) - 0.5 * np.log(2 * np.pi)
        return scipy_logsumexp(log_w, axis=1) - np.log(n)  # (M,)

    def predict(self, X):
        """X: (M, d)"""
        d = X.shape[1]
        log_p1 = np.zeros(len(X))
        log_p2 = np.zeros(len(X))
        for j in range(d):
            log_p1 += self._log_kde_1d(X[:, j], self.X1[:, j], self.h1[j])
            log_p2 += self._log_kde_1d(X[:, j], self.X2[:, j], self.h2[j])
        log_p1 += np.log(self.prior1)
        log_p2 += np.log(self.prior2)
        return np.where(log_p1 >= log_p2, 1, -1)


# ─────────────────────── 非参数法2：k-NN 密度估计 ───────────────────────

class KNNBayes:
    """
    k-NN 贝叶斯分类器 — 联合训练集多数投票
    在合并的训练集中找 k 个最近邻，按多数类别决策。
    等价于 k-NN 密度估计贝叶斯分类器，且天然保留先验信息。
    """

    def __init__(self, k=7):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def fit(self, features, idx1, idx2):
        # 合并训练集，保留标签
        X1 = features[idx1].copy()
        X2 = features[idx2].copy()
        self.X_train = np.vstack([X1, X2])
        self.y_train = np.concatenate([
            np.ones(len(X1), dtype=np.int8),
            -np.ones(len(X2), dtype=np.int8)
        ])
        return self

    def predict(self, X):
        """分批计算距离，避免大内存矩阵"""
        if len(self.X_train) == 0:
            raise ValueError("training data must not be empty")
        k = min(self.k, len(self.X_train))
        batch = 512
        pred = np.empty(len(X), dtype=np.int8)
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]                             # (B, d)
            diff = xb[:, None, :] - self.X_train[None, :, :]  # (B, N, d)
            dists = np.sum(diff ** 2, axis=2)               # (B, N) 省去sqrt
            idx = np.argpartition(dists, k - 1, axis=1)[:, :k]  # (B, k)
            neighbors = self.y_train[idx]                   # (B, k)
            pred[i:i + batch] = np.where(neighbors.sum(axis=1) >= 0, 1, -1).astype(np.int8)
        return pred


# ─────────────────────── 图像分割 ───────────────────────

def apply_mask_and_classify(img_path, mask, classifier, mode='color'):
    """
    在掩码区域内提取特征并分类；mask=None 时对全图分类。
    mode: 'gray' 或 'color'
    返回: img_rgb, img_gray, result(H,W uint8, 255=鱼体)
    """
    img_rgb = np.array(Image.open(img_path).convert('RGB'))
    img_gray = np.array(Image.open(img_path).convert('L'))
    H, W = img_rgb.shape[:2]

    if mask is None:
        # 全图分类
        mask_resized = np.ones((H, W), dtype=bool)
    else:
        # 将 mask 尺寸对齐到图像
        if mask.shape != (H, W):
            from PIL import Image as PILImage
            mask_resized = np.array(
                PILImage.fromarray(mask.astype(np.uint8)).resize(
                    (W, H), PILImage.NEAREST)
            ).astype(bool)
        else:
            mask_resized = mask

    ys, xs = np.where(mask_resized)

    # 训练数据已归一化到 [0,1]，图像像素需同步除以 255
    if mode == 'gray':
        feats = img_gray[ys, xs].reshape(-1, 1).astype(np.float64) / 255.0
    else:
        R    = img_rgb[ys, xs, 0].astype(np.float64) / 255.0
        G    = img_rgb[ys, xs, 1].astype(np.float64) / 255.0
        B    = img_rgb[ys, xs, 2].astype(np.float64) / 255.0
        gray = img_gray[ys, xs].astype(np.float64)   / 255.0
        feats = np.column_stack([gray, R, G, B])

    pred = classifier.predict(feats)

    result = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    result[ys, xs] = (pred == 1).astype(np.uint8) * 255

    # 形态学后处理：先闭运算填空洞，再开运算去噪点
    from scipy.ndimage import binary_closing, binary_opening
    struct = np.ones((5, 5), dtype=bool)
    result_bool = result.astype(bool)
    result_bool = binary_closing(result_bool, structure=struct, iterations=2)
    result_bool = binary_opening(result_bool, structure=struct, iterations=1)
    result = result_bool.astype(np.uint8) * 255

    return img_rgb, img_gray, result


# ─────────────────────── 可视化 ───────────────────────

def visualize_task1(img_rgb, img_gray, results_dict, save_path=None):
    """可视化任务1的多种方法对比"""
    methods = list(results_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))

    # 第一行：灰度图结果
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('原始灰度图')
    axes[0, 0].axis('off')

    # 第二行：彩色图结果
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title('原始彩色图')
    axes[1, 0].axis('off')

    for i, method in enumerate(methods):
        gray_res, color_res = results_dict[method]

        axes[0, i + 1].imshow(gray_res, cmap='gray')
        axes[0, i + 1].set_title(f'{method}\n(灰度)')
        axes[0, i + 1].axis('off')

        # 叠加显示
        overlay = img_rgb.copy()
        overlay[color_res == 255] = [255, 100, 0]
        axes[1, i + 1].imshow(overlay)
        axes[1, i + 1].set_title(f'{method}\n(彩色叠加)')
        axes[1, i + 1].axis('off')

    plt.suptitle('309.bmp — Nemo 鱼像素分割结果对比', fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.show()


def extract_color_regions(img_rgb, fish_mask, orig_mask):
    """
    在鱼体掩码区域内分离橙色和白色像素。
    fish_mask : bool (H,W)  — 贝叶斯分类器输出的鱼体像素
    orig_mask : bool (H,W)  — Mask.mat 对齐后的原始区域（含白条纹）
    返回: orange_img, white_img, overlay_img
    """
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)
    brightness = (R + G + B) / 3
    saturation = (np.max(img_rgb, axis=2).astype(float)
                  - np.min(img_rgb, axis=2).astype(float))

    orange_mask = fish_mask & (R > 150) & (G < 170) & (B < 110)
    white_mask  = orig_mask & (brightness > 130) & (saturation < 65) & ~orange_mask

    orange_img = np.zeros_like(img_rgb)
    orange_img[orange_mask] = img_rgb[orange_mask]

    white_img = np.zeros_like(img_rgb)
    white_img[white_mask] = img_rgb[white_mask]

    overlay = img_rgb.copy()
    overlay[orange_mask] = [255, 120,   0]
    overlay[white_mask]  = [  0, 220, 220]

    return orange_img, white_img, overlay


def visualize_task2(images_info, save_path=None):
    """
    可视化任务2：4行 × n列
    行0: 原图  行1: 橙色区域  行2: 白色区域  行3: 叠加对比
    images_info: list of (name, img_rgb, orange_img, white_img, overlay_img)
    """
    n = len(images_info)
    row_labels = ['原图', '橙色区域', '白色区域', '叠加对比']
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 16))

    for i, (name, img_rgb, orange_img, white_img, overlay_img) in enumerate(images_info):
        panels = [img_rgb, orange_img, white_img, overlay_img]
        for row, (panel, label) in enumerate(zip(panels, row_labels)):
            axes[row, i].imshow(panel)
            axes[row, i].set_title(f'{name}\n{label}', fontsize=10)
            axes[row, i].axis('off')

    plt.suptitle('彩色图像 Nemo 鱼分割结果 — 橙色/白色分离 + 叠加对比 (高斯贝叶斯)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.show()


def plot_feature_distribution(features, idx1, idx2):
    """绘制特征分布直方图"""
    feat_names = ['灰度', 'R', 'G', 'B']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for j, name in enumerate(feat_names):
        f1 = features[idx1, j]
        f2 = features[idx2, j]
        axes[j].hist(f1, bins=40, alpha=0.6, color='orange', label='类别1(鱼体)', density=True)
        axes[j].hist(f2, bins=40, alpha=0.6, color='blue', label='类别2(背景)', density=True)
        axes[j].set_title(f'{name} 特征分布')
        axes[j].set_xlabel('像素值')
        axes[j].legend(fontsize=8)

    plt.suptitle('训练样本特征分布', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.show()


# ─────────────────────── 主流程 ───────────────────────

def main():
    import argparse
    global DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Nemo pixel-classification lab; data is not distributed")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/nemo")
    args = parser.parse_args()
    DATA_DIR, OUTPUT_DIR = args.data_dir, args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 55)
    print("  基于贝叶斯决策的 Nemo 鱼像素点分类")
    print("=" * 55)

    # ── 加载数据 ──
    print("\n[1] 加载数据...")
    features, labels, idx1, idx2, mask = load_data()

    # ── 特征分布可视化 ──
    print("\n[2] 绘制特征分布...")
    plot_feature_distribution(features, idx1, idx2)

    # ── 训练分类器 ──
    print("\n[3] 训练分类器...")

    gray_feats = features[:, 0:1]   # 仅灰度列 (N,1)

    clf_gauss_gray  = GaussianBayes1D().fit(features, idx1, idx2)
    print("  高斯(灰度) 训练完成")
    clf_parzen_gray = ParzenBayes().fit(gray_feats, idx1, idx2)
    print("  Parzen窗(灰度) 训练完成")
    clf_knn_gray    = KNNBayes(k=7).fit(gray_feats, idx1, idx2)
    print("  k-NN(灰度) 训练完成")

    clf_gauss_color  = GaussianBayes().fit(features, idx1, idx2)
    print("  高斯(彩色4D) 训练完成")
    clf_parzen_color = ParzenBayes().fit(features, idx1, idx2)
    print("  Parzen窗/KDE(彩色4D) 训练完成")
    clf_knn_color    = KNNBayes(k=7).fit(features, idx1, idx2)
    print("  k-NN(彩色4D) 训练完成")

    # ── 任务1：309.bmp ──
    print("\n[4] 任务1: 处理 309.bmp ...")
    img_path_309 = os.path.join(DATA_DIR, "309.bmp")

    results_309 = {}
    methods_clfs = [
        ("高斯(参数)",    clf_gauss_gray,  clf_gauss_color),
        ("Parzen窗(非参数)", clf_parzen_gray, clf_parzen_color),
        ("k-NN(非参数)", clf_knn_gray,    clf_knn_color),
    ]

    for method_name, clf_g, clf_c in methods_clfs:
        print(f"  [{method_name}] 分割中...")

        img_rgb, img_gray, gray_result = apply_mask_and_classify(
            img_path_309, mask, clf_g, mode='gray')

        _, _, color_result = apply_mask_and_classify(
            img_path_309, mask, clf_c, mode='color')

        results_309[method_name] = (gray_result, color_result)
        print(f"    鱼体像素数: 灰度={gray_result.sum()//255}, 彩色={color_result.sum()//255}")

    # 可视化任务1
    print("\n[5] 可视化任务1结果...")
    save1 = os.path.join(OUTPUT_DIR, "result_309_comparison.png")
    visualize_task1(img_rgb, img_gray, results_309, save_path=save1)

    # ── 任务2：其他4张图像 ──
    print("\n[6] 任务2: 处理其余4张图像...")
    other_images = ["311.bmp", "313.bmp", "315.bmp", "317.bmp"]
    task2_results = []

    for fname in other_images:
        img_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(img_path):
            print(f"  {fname} 不存在，跳过")
            continue
        print(f"  处理 {fname}...")
        img_rgb_i, _, color_result_i = apply_mask_and_classify(
            img_path, mask, clf_gauss_color, mode='color')

        # 将 Mask.mat 对齐到当前图像尺寸
        Hi, Wi = img_rgb_i.shape[:2]
        if mask.shape != (Hi, Wi):
            from PIL import Image as PILImage
            orig_mask_i = np.array(
                PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (Wi, Hi), PILImage.NEAREST)
            ).astype(bool)
        else:
            orig_mask_i = mask.astype(bool)

        fish_mask_i = color_result_i == 255
        orange_i, white_i, overlay_i = extract_color_regions(
            img_rgb_i, fish_mask_i, orig_mask_i)
        task2_results.append(
            (fname.replace('.bmp', ''), img_rgb_i, orange_i, white_i, overlay_i))
        print(f"    鱼体像素数: {color_result_i.sum()//255}")

    if task2_results:
        print("\n[7] 可视化任务2结果...")
        save2 = os.path.join(OUTPUT_DIR, "result_task2_others.png")
        visualize_task2(task2_results, save_path=save2)

    # ── 保存309.bmp高质量分割图（原图 | 分割掩码 | 橙色 | 白色 | 叠加）──
    print("\n[8] 保存309.bmp高斯彩色分割结果...")
    _, best_result = results_309["高斯(参数)"]
    fish_mask = best_result == 255

    H309, W309 = img_rgb.shape[:2]
    if mask.shape != (H309, W309):
        from PIL import Image as PILImage
        orig_mask_309 = np.array(
            PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                (W309, H309), PILImage.NEAREST)
        ).astype(bool)
    else:
        orig_mask_309 = mask.astype(bool)

    orange_img, white_img, overlay_final = extract_color_regions(
        img_rgb, fish_mask, orig_mask_309)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['原图',  '橙色区域', '白色区域', '叠加对比']
    imgs   = [img_rgb,  orange_img, white_img, overlay_final]
    cmaps  = [None, None, None, None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('309.bmp — 高斯贝叶斯彩色分割（橙/白分离 + 叠加对比）', fontsize=13)
    plt.tight_layout()
    save3 = os.path.join(OUTPUT_DIR, "result_309_best.png")
    plt.savefig(save3, dpi=150, bbox_inches='tight')
    print(f"  已保存: {save3}")
    plt.show()

    print("\n完成！")


if __name__ == "__main__":
    main()
