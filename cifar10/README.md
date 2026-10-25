# CIFAR-10：ResNet 与 VGG 课程实验

课程项目的可运行源码与历史实验记录。主入口支持 ResNet18、ResNet34、VGG16-BN，包含训练/测试、主要层尺寸表、每类准确率和混淆矩阵导出。历史最佳测试准确率为 ResNet18 90.30%、ResNet34 89.70%、VGG16-BN 88.19%；证据强度与 epoch 口径见[历史结果](docs/RESULTS.md)。本轮只执行合成输入的 CPU 检查，没有重新训练 CIFAR-10。

## 文件

- [cifar10_multi.py](cifar10_multi.py)：原 `submit_minimal` 三模型整合入口，原字节保留。
- [experiments](experiments)：4 个较早脚本，保留模型比较、训练日志、FakeData 调试的实现过程。
- [results/historical](results/historical)：历史 CSV 和脱敏 stdout 日志。
- [方法与结构](docs/METHOD.md)、[技术来源和贡献范围](docs/PROVENANCE.md)、[验证记录](docs/VALIDATION.md)。
- [SOURCE_HASHES.md](SOURCE_HASHES.md)：所有收录的原源码/CSV 的 SHA-256。

## 安装与离线检查

在本组件目录运行。使用独立 Python 环境，按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择匹配的 torch/torchvision；CPU 可使用：

```powershell
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install matplotlib
python cifar10_multi.py --help
python tools/smoke_test.py
```

本轮验证环境为 Python 3.13、torch 2.12.0+cpu、torchvision 0.27.0+cpu。matplotlib 仅用于可选 PNG；CSV 导出不依赖它。smoke 仅创建随机张量，三个模型各训练一个 batch、检查前向/梯度/参数更新/混淆矩阵，并检查较早三个模型前向。输出准确率无实际分类意义。

## 获取数据和运行正式实验

数据从 [CIFAR 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)获取。主入口调用 [torchvision.datasets.CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) 的 `download=True`，首次运行会下载官方 Python 数据集；已有完整数据时校验后复用。请遵守数据提供方的使用条件。本仓库不分发数据、预训练权重或课程报告原件。

```powershell
python cifar10_multi.py --model resnet18 --epochs 20 --batch-size 128 --num-workers 0 --data-dir ./data --confusion-dir ./confusion_matrix/resnet18
python cifar10_multi.py --model resnet34 --epochs 20 --batch-size 128 --num-workers 0 --data-dir ./data --confusion-dir ./confusion_matrix/resnet34
python cifar10_multi.py --model vgg16 --epochs 20 --batch-size 128 --num-workers 0 --data-dir ./data --confusion-dir ./confusion_matrix/vgg16
```

以上为复现实验命令，本轮未执行。入口自动选择可用 CUDA 或 CPU，固定 seed=42；输出逐 epoch 指标、最佳测试准确率，恢复内存中的最佳状态后导出每类准确率与混淆矩阵。它不持久保存 checkpoint 或 epoch CSV。需要逐 epoch CSV 可参考历史 VGG16/ResNet34 脚本的 `--log-path`，需要 FakeData 与 batch 上限可使用历史 ResNet18 demo。历史结果来自较早脚本，不能把它们当作整合入口的重新训练成绩。

注意：`experiments/cifar10_homework.py` 在模块顶层创建 CIFAR-10 数据集，导入或运行会尝试下载；它仅作早期实现档案，不用于 smoke。原脚本保留 `KMP_DUPLICATE_LIB_OK` 兼容处理；本机直接裸导入 torch 曾触发 OpenMP 重复运行库错误，验证采用原入口的导入顺序。跨平台依赖与完整训练仍需在目标环境确认。
