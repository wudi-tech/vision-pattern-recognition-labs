# 模式识别课程实验归档

保留 9 份课程 Python 源码，覆盖高斯贝叶斯决策面、像素分类、二维 SVM 与 19×19 人脸/非人脸图像块分类。最后一项是二分类检测练习，不是人员身份识别系统。本组件用于展示课程实现、特征实验与验证边界。

| 范围 | 入口与内容 |
| --- | --- |
| 贝叶斯决策 | [决策面](work1/bayes_decision_boundary1.py)：相同/不同协方差、先验变化 |
| 像素分类 | [Nemo 实验](work2/bayes_nemo.py)：灰度/四维颜色特征，高斯、逐维 KDE 与 kNN，掩码和形态学处理 |
| 二维 SVM | [合成数据生成](work3/task3_2_1v0.py)、[核与缩放可视化](work3/task3_2_2v1.py) |
| 图像块 SVM | [像素](work3/test3_1.py)、[LBP](work3/test3_LBP.py)、[像素+LBP](work3/test3_4.py)、[像素+统计+HOG](work3/test3_5.py)、[复用工具](work3/test3_svm_utils.py) |

本轮只做小规模合成 CPU 检查；未重新运行课程数据评测。历史记录中的 0.855、0.860、0.812、0.817、0.888 缺少完整协议和测试标签，不能解释成已复现的精度或受控改进。具体边界见[实验记录](docs/EXPERIMENTS.md)。

安装依赖并运行不依赖课程数据的检查：

```powershell
python -m pip install -r requirements.txt
python -B tests/smoke.py
```

检查直接调用实际实现和 LIBSVM，使用随机合成数组及临时生成的图像。它不下载数据，不读取课程 MAT/电影图像，不进行完整网格搜索。当前验证环境及断言范围见[验证说明](docs/VALIDATION.md)与[机器记录](docs/VALIDATION.json)。

历史入口需要自己合法取得的输入：

```powershell
python -B work2/bayes_nemo.py --data-dir ./data/nemo --output-dir ./outputs/nemo
python -B work3/test3_5.py --data-dir ./data/face --output-dir ./outputs/face
```

运行历史人脸入口会启动较大的网格搜索；日常检查使用 smoke。其余三个图像块入口也接受相同目录参数。输入格式见[数据契约](docs/DATA_CONTRACT.md)。二维生成器和可视化程序沿用当前工作目录读写与交互绘图行为：在独立输出目录中运行生成器，再运行可视化程序；字体显示取决于本机字体。上述历史绘图主流程本轮未运行。

源码来自个人保存的课程工程及其中的多轮辅助编程迭代。标准算法、LIBSVM 求解器和课程材料不属于个人原创成果；不能根据归档推断每行代码均由本人独立编写。[来源与权利](docs/PROVENANCE.md)说明贡献及分发边界，[源码清单](docs/SOURCE_MANIFEST.json)记录原始/公开 SHA-256 及改动。
