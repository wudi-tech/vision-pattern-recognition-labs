# 第三方方法与材料说明

本项目依赖 NumPy 与 OpenCV Python 的安装包，不在本组件内复制其源码或二进制。它们各自的许可证及依赖声明适用于相应第三方软件；本说明不代替其许可证。

- SIFT 由 David G. Lowe 提出，本项目调用 OpenCV 的实现：[Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)。
- OpenCV 官方示例说明了 SIFT、FLANN、比值筛选及 RANSAC 单应估计的标准组合：[Feature Matching + Homography to find Objects](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)。本项目的相同流程和参数不作为自创方法。
- [OpenCV 项目](https://github.com/opencv/opencv)、[opencv-python 打包项目](https://github.com/opencv/opencv-python)、[NumPy 项目](https://github.com/numpy/numpy) 的授权以各项目许可证和安装包声明为准。

全景参考系组织、掩码/权重累计、裁剪和实验输出属于本课程实现的项目代码，包含 AI 辅助。现有证据不支持将整个系统描述成完全从零独立研发。未查明来源和公开分发许可的历史照片、课程 PDF 与个人报告没有随附。

本次没有给不明来源材料重新授予许可，也没有新设覆盖全部历史素材的组件许可证。若上层仓库采用许可证，应避免将已排除的本地材料误认为在其授权范围。
