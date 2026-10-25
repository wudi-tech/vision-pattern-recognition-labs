# 本轮验证记录

验证日期：2026-10-25。运行环境：Windows、Python 3.13（Anaconda 解释器）、torch 2.12.0+cpu、torchvision 0.27.0+cpu。使用本机已安装包，未安装或下载模型/数据。

在组件目录执行：

```powershell
python -B cifar10_multi.py --help
python -B tools/smoke_test.py
```

实际离线 smoke 退出码为 0，输出 `SYNTHETIC_CHECK_PASS`：

| 模型 | 前向输出 | 优化器步数 | 参数更新 | 混淆矩阵样本总数 | 参数量 |
| --- | --- | ---: | --- | ---: | ---: |
| ResNet18 | `(2,10)` | 1 | 已确认 | 10 | 11,173,962 |
| ResNet34 | `(2,10)` | 1 | 已确认 | 10 | 21,282,122 |
| VGG16-BN | `(2,10)` | 1 | 已确认 | 10 | 15,249,354 |

检查覆盖：有限 logits/损失/梯度、首个参数更新、结构尺寸表、每类计数、原始和百分比混淆矩阵 CSV 的 11×11 格式；另对三个历史比较模型实际前向，输出均为 `(2,10)`。PNG 导出是原程序中的可选功能，异常会被吞掉，因此 CSV 通过不能当作 PNG 绘制成功的证明。

5 个历史/最终 Python 文件与新增 smoke 脚本均做 AST 检查。早期 `cifar10_homework.py` 因导入即触发真实数据下载，仅做静态检查。收录源码/CSV 与原文件 hash 比对一致；stdout 只替换机器路径；公开文件扫描身份信息、绝对机器路径、常见密钥模式和 Markdown 本地链接。

未执行完整 CIFAR-10 数据加载、正式训练、真实数据准确率复核、GPU 性能测试、不同平台兼容测试；合成张量上的准确率不作为实验结果。ResNet18 本地 checkpoint 仅 `weights_only=True` 只读检查元信息，未发布、未用真实数据评估。原程序保留 Windows OpenMP 兼容项，本机裸 `import torch,torchvision` 曾失败，按原入口顺序导入后本轮检查通过。
