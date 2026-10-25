# 技术来源与贡献范围

这是使用既有 VGG/ResNet 方法完成的课程分类实验归档。可核验的项目工作包括 CIFAR 图像尺寸适配、训练/测试流程组织、参数配置、三种架构的实验比较、尺寸表与结果导出，以及最终三模型入口整合。这些工程内容不代表标准网络架构、残差学习、数据集、PyTorch 或 torchvision 的原创贡献；当前材料也不足以逐行认定所有历史代码的独立原创来源。整理过程中有 AI 辅助阅读、文档与离线检查。

主要方法和依赖：

- He 等，[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)：ResNet 与残差连接。
- Simonyan 与 Zisserman，[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)：VGG 结构。
- Krizhevsky、Nair、Hinton 的 [CIFAR 数据集页面](https://www.cs.toronto.edu/~kriz/cifar.html)：数据与引用要求。
- [PyTorch](https://pytorch.org/) 和 [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)：张量、神经网络、优化器、数据加载和增强 API。当前收录脚本自行组合 `nn.Module`，没有调用 `torchvision.models` 或下载预训练模型。

本次整理保留 5 个 Python 原文件和 4 个 CSV 的原字节，只重排公开路径。两份 stdout 仅替换机器绝对输出路径。新增加公开说明、hash 清单和 CPU smoke 脚本；没有改写模型算法、伪造或补跑历史指标。源码详细对应见 [SHA-256 清单](../SOURCE_HASHES.md)。

课程要求 PDF、含身份信息的提交 ZIP/报告、编辑器设置、CIFAR 数据、checkpoint 以及无关图片不公开。原工程只读保留。第三方方法与依赖各自的版权/许可证不因本归档改变；本组件没有证据支持对全部历史源码额外授予统一开源许可证。
