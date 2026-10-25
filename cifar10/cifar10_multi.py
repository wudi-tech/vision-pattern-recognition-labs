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


CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir, batch_size, num_workers):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class VGG16CIFAR(nn.Module):
    """VGG16: 13 convolution layers and 3 fully connected layers."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = self._make_block(3, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 3)
        self.block4 = self._make_block(256, 512, 3)
        self.block5 = self._make_block(512, 512, 3)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_block(self, in_channels, out_channels, num_convs):
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock: F(x) + shortcut(x)."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNetCIFAR(nn.Module):
    """CIFAR-10 ResNet: 3x3 stem, no initial max pooling."""

    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(name):
    if name == "vgg16":
        return VGG16CIFAR()
    if name == "resnet18":
        return ResNetCIFAR([2, 2, 2, 2])
    if name == "resnet34":
        return ResNetCIFAR([3, 4, 6, 3])
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def print_shape_table(model, model_name, batch_size, device):
    model.eval()
    x = torch.zeros(batch_size, 3, 32, 32, device=device)
    rows = []

    if model_name == "vgg16":
        modules = [("block1", model.block1), ("block2", model.block2), ("block3", model.block3),
                   ("block4", model.block4), ("block5", model.block5)]
        for name, module in modules:
            in_shape = tuple(x.shape)
            x = module(x)
            rows.append((name, in_shape, tuple(x.shape)))
        in_shape = tuple(x.shape)
        x = torch.flatten(x, 1)
        rows.append(("flatten", in_shape, tuple(x.shape)))
        in_shape = tuple(x.shape)
        x = model.classifier(x)
        rows.append(("classifier", in_shape, tuple(x.shape)))
    else:
        modules = [("stem", model.stem), ("layer1", model.layer1), ("layer2", model.layer2),
                   ("layer3", model.layer3), ("layer4", model.layer4), ("avgpool", model.avgpool)]
        for name, module in modules:
            in_shape = tuple(x.shape)
            x = module(x)
            rows.append((name, in_shape, tuple(x.shape)))
        in_shape = tuple(x.shape)
        x = torch.flatten(x, 1)
        rows.append(("flatten", in_shape, tuple(x.shape)))
        in_shape = tuple(x.shape)
        x = model.fc(x)
        rows.append(("fc", in_shape, tuple(x.shape)))

    print("\nModel structure and tensor sizes:")
    print(f"{'Layer':12s} {'Input':24s} {'Output':24s}")
    for name, in_shape, out_shape in rows:
        print(f"{name:12s} {str(in_shape):24s} {str(out_shape):24s}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_num = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_num += labels.size(0)
    return total_loss / total_num, total_correct / total_num * 100


@torch.no_grad()
def test(model, loader, criterion, device, show_class_acc=False):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion = [[0 for _ in range(10)] for _ in range(10)]

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

        for label, pred in zip(labels, preds):
            idx = label.item()
            pred_idx = pred.item()
            confusion[idx][pred_idx] += 1
            class_total[idx] += 1
            class_correct[idx] += int(idx == pred_idx)

    if show_class_acc:
        print("\nPer-class accuracy:")
        for i, name in enumerate(CLASSES):
            acc = class_correct[i] / class_total[i] * 100
            print(f"{name:>5s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    return total_loss / total_num, total_correct / total_num * 100, confusion


def save_confusion_matrix(confusion, model_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{model_name}_confusion_matrix.csv"
    norm_path = output_dir / f"{model_name}_confusion_matrix_percent.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(CLASSES))
        for class_name, row in zip(CLASSES, confusion):
            writer.writerow([class_name] + row)

    with norm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(CLASSES))
        for class_name, row in zip(CLASSES, confusion):
            total = sum(row)
            percent_row = [round(value / total * 100, 2) if total else 0.0 for value in row]
            writer.writerow([class_name] + percent_row)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(10), CLASSES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{model_name} confusion matrix")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    print(f"Confusion matrix saved to: {raw_path}")
    print(f"Normalized confusion matrix saved to: {norm_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "resnet34", "vgg16"], default="resnet18")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--confusion-dir", default="./confusion_matrix")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(args.model).to(device)
    print_shape_table(model, args.model, args.batch_size, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1,
    )

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _ = test(model, test_loader, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f}, test acc {test_acc:.2f}%"
        )

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    _, _, confusion = test(model, test_loader, criterion, device, show_class_acc=True)
    save_confusion_matrix(confusion, args.model, args.confusion_dir)


if __name__ == "__main__":
    main()
