"""
Computer Vision Homework 2 Demo
ResNet18 on CIFAR-10 with PyTorch

This file is written as both runnable code and report material.
The Chinese comments mark which part corresponds to which homework flow:
1. 数据准备
2. ResNet 模型搭建
3. 训练阶段
4. 测试阶段
5. 结构与维度说明

Recommended quick run:
    python resnet_cifar10_demo.py --epochs 1 --max-train-batches 20 --max-test-batches 10

Recommended formal run:
    python resnet_cifar10_demo.py --epochs 20 --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# 部分 Windows + PyTorch/NumPy 环境会因为 OpenMP 动态库重复加载而在 import torch
# 阶段报 libiomp5md.dll 错误。这里设置为本脚本内的兼容兜底，保证 demo 能正常运行。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# CIFAR-10 的 10 个类别名称，测试阶段可以用来输出每一类的准确率。
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_seed(seed: int) -> None:
    """固定随机种子，便于多次实验结果尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1. 数据准备：CIFAR-10
# =========================
def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    use_fake_data: bool,
) -> tuple[DataLoader, DataLoader]:
    """
    作业中需要说明：
    - CIFAR-10 图片大小是 3x32x32，类别数是 10。
    - 训练集使用随机裁剪和水平翻转做数据增强。
    - 测试集不做随机增强，只做 ToTensor 和 Normalize。
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    if use_fake_data:
        # 这个分支只用于快速检查代码能否跑通，不用于正式实验结果。
        # FakeData 生成的样本尺寸同样是 (3, 32, 32)，类别数同样是 10。
        train_set = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        test_set = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
    else:
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=test_transform,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# =====================================================
# 2. ResNet 模型搭建：重点说明 BasicBlock 和整体网络
# =====================================================
def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """
    ResNet 的基础卷积：3x3 卷积。

    padding=1 可以保持特征图高宽不变；如果 stride=2，则高宽减半。
    例如输入 (B, 64, 32, 32)，stride=2 输出 (B, 128, 16, 16)。
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    ResNet18/ResNet34 使用的基础残差块。

    这部分是文档里最应该讲清楚的内容：
    - 主分支 residual branch：
      Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    - 捷径分支 shortcut branch：
      如果输入输出维度一致，直接用 Identity。
      如果通道数变化或高宽减半，用 1x1 卷积调整维度。
    - 最后把主分支和捷径分支相加，再经过 ReLU。

    残差连接的核心公式：
        y = F(x) + x
    当维度不一致时：
        y = F(x) + W_s x
    其中 W_s 是 1x1 卷积，用来匹配维度。
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # 主分支第 1 层：可能负责下采样。
        # stride=1 时高宽不变；stride=2 时高宽减半。
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 主分支第 2 层：只提取特征，不改变高宽。
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径分支：默认直接传递输入。
        # 如果 stride != 1，说明主分支让特征图高宽减半；
        # 如果 in_channels != out_channels，说明通道数变化；
        # 这两种情况都必须用 1x1 卷积把 shortcut 调整到和主分支一样的尺寸。
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支 F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 捷径分支 shortcut(x)，然后执行残差相加。
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    适配 CIFAR-10 的 ResNet。

    注意：ImageNet 版 ResNet 的输入通常是 224x224，所以开头会使用：
        7x7 Conv, stride=2 + MaxPool
    但 CIFAR-10 输入只有 32x32，如果一开始就大幅下采样，会丢失太多细节。
    因此这里采用 CIFAR 常用写法：
        3x3 Conv, stride=1，不使用开头的 MaxPool。

    ResNet18 的四个 stage：
        layer1: 2 个 BasicBlock，通道 64，高宽 32x32
        layer2: 2 个 BasicBlock，通道 128，高宽 16x16
        layer3: 2 个 BasicBlock，通道 256，高宽 8x8
        layer4: 2 个 BasicBlock，通道 512，高宽 4x4
    """

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = 64

        # stem：把原始 RGB 图片从 3 通道映射到 64 通道。
        # 输入 (B, 3, 32, 32)，输出 (B, 64, 32, 32)。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 四个残差 stage。每个 stage 由若干 BasicBlock 组成。
        # layer1 不下采样；layer2/layer3/layer4 的第一个 block 使用 stride=2 下采样。
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=num_blocks[3], stride=2)

        # 全局平均池化把每个通道的 4x4 特征图压缩成 1x1。
        # 输入 (B, 512, 4, 4)，输出 (B, 512, 1, 1)。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层，把 512 维特征映射到 10 个类别。
        # 输入 (B, 512)，输出 (B, 10)。
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        搭建一个 ResNet stage。

        以 layer2 为例：
        - 第一个 BasicBlock 使用 stride=2：
          输入 (B, 64, 32, 32)，输出 (B, 128, 16, 16)。
        - 第二个 BasicBlock 使用 stride=1：
          输入 (B, 128, 16, 16)，输出 (B, 128, 16, 16)。

        因此，每个 stage 通常只有第一个 block 负责改变尺寸，
        后续 block 只在相同尺寸上继续提取特征。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for block_stride in strides:
            layers.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播流程，也就是文档结构图中的主线。
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar10(num_classes: int = 10) -> ResNetCIFAR:
    """
    创建 ResNet18。

    [2, 2, 2, 2] 表示四个 stage 分别有 2 个 BasicBlock：
        stem + layer1(2) + layer2(2) + layer3(2) + layer4(2) + avgpool + fc
    这就是 ResNet18 的主体结构。
    """
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ========================================
# 3. 结构与维度说明：给实验文档直接使用
# ========================================
@torch.no_grad()
def print_shape_table(model: ResNetCIFAR, batch_size: int, device: torch.device) -> None:
    """
    打印各主要层输入/输出维度。

    作业文档可以直接采用这张表，B 可以替换成具体 batch size。
    默认 CIFAR-10 输入是 (B, 3, 32, 32)。
    """
    model.eval()
    x = torch.zeros(batch_size, 3, 32, 32, device=device)
    rows = []

    def run_and_record(name: str, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        input_shape = tuple(value.shape)
        output = module(value)
        output_shape = tuple(output.shape)
        rows.append((name, input_shape, output_shape))
        return output

    x = run_and_record("stem: 3x3 Conv + BN + ReLU", model.stem, x)
    x = run_and_record("layer1: 2 BasicBlocks", model.layer1, x)
    x = run_and_record("layer2: 2 BasicBlocks, first stride=2", model.layer2, x)
    x = run_and_record("layer3: 2 BasicBlocks, first stride=2", model.layer3, x)
    x = run_and_record("layer4: 2 BasicBlocks, first stride=2", model.layer4, x)
    x = run_and_record("avgpool: AdaptiveAvgPool2d", model.avgpool, x)

    input_shape = tuple(x.shape)
    x = torch.flatten(x, 1)
    rows.append(("flatten", input_shape, tuple(x.shape)))

    input_shape = tuple(x.shape)
    x = model.fc(x)
    rows.append(("fc: Linear classifier", input_shape, tuple(x.shape)))

    print("\nModel shape table for report:")
    print("-" * 88)
    print(f"{'Layer':40s} {'Input shape':22s} {'Output shape':22s}")
    print("-" * 88)
    for name, input_shape, output_shape in rows:
        print(f"{name:40s} {str(input_shape):22s} {str(output_shape):22s}")
    print("-" * 88)


# =========================
# 4. 训练阶段
# =========================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int | None,
) -> tuple[float, float]:
    """
    训练流程需要在文档中说明：
    1. 输入一批图片 images 和标签 labels。
    2. model(images) 得到预测 logits。
    3. CrossEntropyLoss 计算分类损失。
    4. loss.backward() 反向传播。
    5. optimizer.step() 更新网络参数。
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples * 100
            print(f"Epoch {epoch:03d} | batch {batch_idx:04d} | loss {avg_loss:.4f} | acc {avg_acc:.2f}%")

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples * 100
    return epoch_loss, epoch_acc


# =========================
# 5. 测试阶段
# =========================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    show_class_accuracy: bool = True,
) -> tuple[float, float]:
    """
    测试流程需要在文档中说明：
    - model.eval() 进入测试模式。
    - torch.no_grad() 关闭梯度计算，节省显存/内存。
    - 只做前向传播，不更新参数。
    - 统计整体准确率，也可以统计每一类准确率。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0 for _ in CLASSES]
    class_total = [0 for _ in CLASSES]

    for batch_idx, (images, labels) in enumerate(test_loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predicted = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        for true_label, pred_label in zip(labels, predicted):
            label_id = true_label.item()
            class_total[label_id] += 1
            if true_label == pred_label:
                class_correct[label_id] += 1

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples * 100

    if show_class_accuracy:
        print("\nPer-class accuracy:")
        for class_name, correct, total in zip(CLASSES, class_correct, class_total):
            if total == 0:
                print(f"{class_name:>5s}: no samples")
            else:
                print(f"{class_name:>5s}: {correct / total * 100:6.2f}% ({correct}/{total})")

    return test_loss, test_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet18 demo for CIFAR-10 homework.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-download", action="store_true", help="Use this if CIFAR-10 is already downloaded.")
    parser.add_argument("--use-fake-data", action="store_true", help="Debug only: run with fake CIFAR-shaped data.")
    parser.add_argument("--save-path", type=str, default=str(Path(__file__).parent / "best_resnet18_cifar10.pth"))
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit training batches for quick demo.")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Limit testing batches for quick demo.")
    parser.add_argument("--no-shape-table", action="store_true", help="Do not print the model shape table.")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
        use_fake_data=args.use_fake_data,
    )

    # 创建 ResNet18-CIFAR 模型。这里是作业中“采用 ResNet 模型”的核心实现。
    model = resnet18_cifar10(num_classes=10).to(device)

    if not args.no_shape_table:
        print_shape_table(model, batch_size=args.batch_size, device=device)

    # 分类任务常用 CrossEntropyLoss。
    criterion = nn.CrossEntropyLoss()

    # ResNet 训练常用 SGD + momentum + weight_decay。
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 学习率调度：训练到后期降低学习率，通常有助于收敛。
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, args.epochs // 2), max(1, args.epochs * 3 // 4)],
        gamma=0.1,
    )

    best_acc = 0.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_test_batches,
            show_class_accuracy=False,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} summary | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": "ResNet18-CIFAR10",
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "classes": CLASSES,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # 最后再完整输出一次每类准确率，方便写实验结果。
    evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_test_batches,
        show_class_accuracy=True,
    )


if __name__ == "__main__":
    main()
