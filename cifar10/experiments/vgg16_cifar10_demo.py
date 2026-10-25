"""
VGG16-CIFAR10 experiment for Computer Vision Homework 2.

Run:
    python vgg16_cifar10_demo.py --epochs 20 --batch-size 128

This script is independent from the ResNet demo. It includes:
1. CIFAR-10 data loading
2. VGG16 model construction
3. Training
4. Testing
5. CSV logging and best-model saving
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    """Prepare CIFAR-10 train/test dataloaders with the same transforms as ResNet18."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


class VGG16CIFAR(nn.Module):
    """
    VGG16 adapted to CIFAR-10.

    VGG16 has 13 convolution layers and 3 fully connected layers.
    The convolution blocks are:
        block1: 2 conv layers, 64 channels
        block2: 2 conv layers, 128 channels
        block3: 3 conv layers, 256 channels
        block4: 3 conv layers, 512 channels
        block5: 3 conv layers, 512 channels

    Each block ends with MaxPool2d(2, 2), so the feature map changes as:
        32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1

    BatchNorm is added after each convolution for stable CIFAR-10 training.
    It does not change the spatial size or channel count.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.block1 = self._make_block(3, 64, num_convs=2)
        self.block2 = self._make_block(64, 128, num_convs=2)
        self.block3 = self._make_block(128, 256, num_convs=3)
        self.block4 = self._make_block(256, 512, num_convs=3)
        self.block5 = self._make_block(512, 512, num_convs=3)

        # CIFAR-10 reaches (B, 512, 1, 1) after five 2x2 max-pooling layers.
        # Therefore the classifier input dimension is 512 instead of 512*7*7
        # used by ImageNet VGG16.
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _make_block(self, in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for conv_idx in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if conv_idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@torch.no_grad()
def print_shape_table(model: VGG16CIFAR, batch_size: int, device: torch.device) -> None:
    model.eval()
    x = torch.zeros(batch_size, 3, 32, 32, device=device)
    rows = []

    for name in ["block1", "block2", "block3", "block4", "block5"]:
        module = getattr(model, name)
        input_shape = tuple(x.shape)
        x = module(x)
        rows.append((name, input_shape, tuple(x.shape)))

    input_shape = tuple(x.shape)
    x = torch.flatten(x, 1)
    rows.append(("flatten", input_shape, tuple(x.shape)))

    input_shape = tuple(x.shape)
    x = model.classifier(x)
    rows.append(("classifier", input_shape, tuple(x.shape)))

    print("\nVGG16-CIFAR10 shape table:")
    print("-" * 78)
    print(f"{'Layer':16s} {'Input shape':28s} {'Output shape':28s}")
    print("-" * 78)
    for name, input_shape, output_shape in rows:
        print(f"{name:16s} {str(input_shape):28s} {str(output_shape):28s}")
    print("-" * 78)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch:03d} | batch {batch_idx:04d} | "
                f"loss {total_loss / total_samples:.4f} | "
                f"acc {total_correct / total_samples * 100:.2f}%"
            )

    return total_loss / total_samples, total_correct / total_samples * 100


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_class_accuracy: bool = False,
) -> tuple[float, float, list[tuple[str, float, int, int]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0 for _ in CLASSES]
    class_total = [0 for _ in CLASSES]

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

    per_class = []
    if show_class_accuracy:
        print("\nPer-class accuracy:")
    for class_name, correct, total in zip(CLASSES, class_correct, class_total):
        acc = correct / total * 100 if total else 0.0
        per_class.append((class_name, acc, correct, total))
        if show_class_accuracy:
            print(f"{class_name:>5s}: {acc:6.2f}% ({correct}/{total})")

    return total_loss / total_samples, total_correct / total_samples * 100, per_class


def write_epoch_log(log_path: Path, rows: list[dict[str, float]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG16-CIFAR10 experiment.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-path", type=str, default=str(Path(__file__).parent / "best_vgg16_cifar10.pth"))
    parser.add_argument("--log-path", type=str, default=str(Path(__file__).parent / "results" / "vgg16_log.csv"))
    parser.add_argument("--no-save-model", action="store_true", help="Do not save model weights; useful for pure experiments.")
    parser.add_argument("--no-shape-table", action="store_true")
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = VGG16CIFAR(num_classes=10).to(device)

    if not args.no_shape_table:
        print_shape_table(model, args.batch_size, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, args.epochs // 2), max(1, args.epochs * 3 // 4)],
        gamma=0.1,
    )

    best_acc = 0.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        write_epoch_log(Path(args.log_path), log_rows)

        print(
            f"Epoch {epoch:03d} summary | "
            f"lr {current_lr:.5f} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                "model": "VGG16-CIFAR10-BN",
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "classes": CLASSES,
                "args": vars(args),
            }
            if args.no_save_model:
                print(f"New best model found: {best_acc:.2f}% (model saving disabled)")
            else:
                try:
                    torch.save(checkpoint, save_path)
                    print(f"Saved best model to: {save_path}")
                except Exception as exc:
                    print(f"Warning: failed to save best model to {save_path}: {exc}")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    _, _, per_class = evaluate(model, test_loader, criterion, device, show_class_accuracy=True)

    per_class_path = Path(args.log_path).with_name("vgg16_per_class.csv")
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy", "correct", "total"])
        writer.writerows(per_class)
    print(f"Saved logs to: {args.log_path}")
    print(f"Saved per-class result to: {per_class_path}")


if __name__ == "__main__":
    main()
