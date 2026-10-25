"""
CIFAR-10 图像识别 —— ResNet18 精简版
满足作业要求：
1) ResNet 模型
2) 包含训练、测试阶段与结果
3) 自动打印各层输入/输出维度
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------- 超参数 ----------
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 固定随机种子 ----------
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------- 数据准备 ----------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------- ResNet 组件 ----------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
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

def resnet18_cifar10():
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2])

# ---------- 打印维度表（用于文档） ----------
@torch.no_grad()
def print_dim_table(model, device):
    x = torch.zeros(1, 3, 32, 32, device=device)
    print("\n各层输入/输出维度：")
    print(f"{'stem':40s} {str(tuple(x.shape)):22s} -> ", end="")
    x = model.stem(x); print(str(tuple(x.shape)))
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        name = f"layer{i+1}"
        print(f"{name:40s} {str(tuple(x.shape)):22s} -> ", end="")
        x = layer(x); print(str(tuple(x.shape)))
    print(f"{'avgpool':40s} {str(tuple(x.shape)):22s} -> ", end="")
    x = model.avgpool(x); print(str(tuple(x.shape)))
    print(f"{'flatten':40s} {str(tuple(x.shape)):22s} -> ", end="")
    x = torch.flatten(x, 1); print(str(tuple(x.shape)))
    print(f"{'fc':40s} {str(tuple(x.shape)):22s} -> ", end="")
    x = model.fc(x); print(str(tuple(x.shape)))
    print()

# ---------- 训练 ----------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total

# ---------- 测试 ----------
@torch.no_grad()
def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total

# ---------- 每类准确率 ----------
@torch.no_grad()
def class_accuracy(model, loader, device):
    model.eval()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = [0.] * 10
    class_total = [0.] * 10
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        for t, p in zip(targets, predicted):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
    print("\n每类准确率：")
    for i in range(10):
        if class_total[i] > 0:
            print(f"{classes[i]:>5s}: {100.*class_correct[i]/class_total[i]:.2f}%")
        else:
            print(f"{classes[i]:>5s}: 无样本")

# ---------- 主程序 ----------
def main():
    model = resnet18_cifar10().to(DEVICE)
    print_dim_table(model, DEVICE)          # 维度表，文档直接使用

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, DEVICE)
        test_loss, test_acc = test_epoch(model, testloader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
              f"test loss {test_loss:.4f} | test acc {test_acc:.2f}%")

    class_accuracy(model, testloader, DEVICE)

if __name__ == "__main__":
    main()