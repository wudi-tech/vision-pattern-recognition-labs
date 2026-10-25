# 方法与结构

训练集采用 `RandomCrop(32, padding=4)`、`RandomHorizontalFlip`、`ToTensor`，按 mean `(0.4914, 0.4822, 0.4465)`、std `(0.2470, 0.2435, 0.2616)` 归一化；测试集只转张量和归一化。整合入口使用 CrossEntropyLoss、SGD（lr 默认 0.1、momentum 0.9、weight decay 5e-4），20 epoch 时 MultiStepLR 在第 10、15 轮后衰减为原来的 0.1。

ResNet 使用适配小图像的 `3×3, stride=1` stem，无最初的 MaxPool；BasicBlock 主分支两层 3×3 卷积，通过 Identity 或 `1×1 Conv+BN` shortcut 相加，再 ReLU。ResNet18 的 block 数为 `[2,2,2,2]`，ResNet34 为 `[3,4,6,3]`。

| ResNet 阶段 | 输出 `(B,C,H,W)` |
| --- | --- |
| 输入 | `(B,3,32,32)` |
| stem / layer1 | `(B,64,32,32)` |
| layer2 | `(B,128,16,16)` |
| layer3 | `(B,256,8,8)` |
| layer4 | `(B,512,4,4)` |
| 全局池化 | `(B,512,1,1)` |
| flatten / fc | `(B,512)` / `(B,10)` |

VGG16-BN 使用 `[2,2,3,3,3]` 个 Conv-BN-ReLU 后接 MaxPool 的卷积块，输出依次为 `(B,64,16,16)`、`(B,128,8,8)`、`(B,256,4,4)`、`(B,512,2,2)`、`(B,512,1,1)`；flatten 后经 `512→512→512→10` 分类器，隐藏层包含 ReLU 和 Dropout(0.5)。这是面向 CIFAR 尺寸的调整，不是 torchvision ImageNet VGG 的原样分类头。

训练每 batch 执行清零梯度、前向、损失、反向和优化器更新。测试使用 `eval()` 和 `no_grad()`；混淆矩阵行是真实类别、列是预测类别，百分比 CSV 按真实类别行归一化。结构尺寸由入口 `print_shape_table` 实际计算。
