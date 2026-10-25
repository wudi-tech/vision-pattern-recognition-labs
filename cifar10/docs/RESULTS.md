# 历史实验结果与证据边界

以下是课程工程已有记录，本轮未重新训练或在真实测试集重新评测。

| 模型 | 历史最佳测试准确率 | 最佳 epoch | 最后 epoch 测试准确率 | 依据 |
| --- | ---: | --- | --- | --- |
| ResNet18 | 90.30% | 未保留 | 原报告记为约 90.26%，无逐 epoch 日志佐证 | 本地 checkpoint `best_acc=90.3` 与原报告；checkpoint 不公开 |
| ResNet34 | 89.70% | 19 / 20 | 89.58%（epoch 20） | [epoch CSV](../results/historical/resnet34_log.csv)、[stdout](../results/historical/resnet34_stdout.log) |
| VGG16-BN | 88.19% | 20 / 20 | 88.19%（epoch 20） | [epoch CSV](../results/historical/vgg16_log.csv)、[stdout](../results/historical/vgg16_stdout.log) |

代码使用 CIFAR-10 官方 `train=True` 的训练集与 `train=False` 的测试集；历史每类 CSV 每类 1000 个样本、共 10000 个，与官方测试集规模相符。未建立单独验证集，而是每个 epoch 观察测试集并据此选最佳状态，因此这些是“按测试集选 epoch 的最佳测试准确率”，不能当作未参与选择的独立泛化估计。三种模型各仅有一组历史记录，无重复种子实验或显著性检验。

VGG16 和 ResNet34 的学习率 CSV 记录 epoch 1–10 为 0.1、11–15 为 0.01、16–20 为 0.001。历史 stdout 记录 CUDA 运行；原报告写 RTX 3060 Laptop GPU，硬件型号未在本轮重跑验证。

历史 [ResNet34 每类 CSV](../results/historical/resnet34_per_class.csv) 的正确数合计 8958，与最后 epoch 的 89.58% 一致；它不是 epoch 19 最佳状态的分类明细。历史 [VGG16 每类 CSV](../results/historical/vgg16_per_class.csv) 合计 8819，与最后/最佳 88.19% 一致。历史比较脚本在结束后直接评估最后模型；最终整合入口则恢复内存中的最佳模型后评估，两种导出口径不同。

ResNet18 checkpoint 本轮以 `torch.load(..., map_location='cpu', weights_only=True)` 只读核对：`model=ResNet18-CIFAR10`、`best_acc=90.3`、`epochs=20`、`batch_size=128`、`seed=42`、`use_fake_data=False`，训练/测试 batch 上限均为 `None`。没有完整逐 epoch 日志，无法确定最佳发生在哪一轮；也未用真实数据验证权重准确率或将权重与最终整合模型互换。

初始化差异也限制直接复现：历史 VGG16 脚本显式执行 Kaiming/BN/Linear 初始化，整合入口使用 PyTorch 层默认初始化。历史结果不归属整合入口的一次新运行。模型排名只描述这些已有记录，不能据此证明某种网络结构普遍优于另一种。
