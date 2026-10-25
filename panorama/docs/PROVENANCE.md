# 来源、版本与贡献边界

原始项目含四份 Python、方法报告、代码讲解、课程任务文件、个人报告、照片、结果图和两份历史提交 ZIP。公开组件保留全部四份 Python 的原始字节；个人资料、原图和重复 ZIP 留在本地，不复制进仓库。

| 原源码 | 公开位置 | 定位 |
|---|---|---|
| `CV_work1.py` | `../CV_work1.py` | 现存最终版，与两个历史提交 ZIP 内的同名源码 SHA-256 一致 |
| `CV_work1_rest.py` | `../reference/CV_work1_rest.py` | 基础三图平均融合版本 |
| `CV_work1_0.py` | `../reference/CV_work1_0.py` | 羽化与最大有效矩形裁剪版本 |
| `Pasted_code_commented.py` | `../reference/Pasted_code_commented.py` | 注释学习版，保留其历史输出名 |

具体字节哈希及匿名归档标识见 [source_manifest.json](source_manifest.json)。哈希可以证明文件身份，不能单独证明所有代码均由一个人独立编写。

原聊天记录显示本项目曾接受 AI 辅助进行方法讲解、代码注释、融合/裁剪方案调整、提交整理、概率溢出和无特征检查等工作。本归档不把 AI 辅助生成或调整的部分表述为独立算法发明。用户完成课程实验与实现整合的具体逐行原创边界，现有文件不足以完全追溯。

SIFT、FLANN、RANSAC、单应估计、距离变换、图像缩放和投影由 OpenCV 提供；项目级代码组织了三图参考系、掩码融合、裁剪和统计输出。基础方法归属见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

旧方法报告仍引用当前目录不存在的 `CV_work1_submit.py` 和 `submit_result_*.jpg`，不能据此凭空恢复一个“提交源码”。本归档选用可核对的 `CV_work1.py`，历史统计明确标成旧报告记载。

本次新增 `run_panorama.py` 只适配路径、输入大小与输出目录，随后调用原核心 `main()`；`tests/smoke.py`、合成图片及本组公开说明是归档阶段新增，不属于历史课程提交原件。原项目的四份 Markdown 未直接复制：其中存在版本混用、失效资源引用及过强结论；可核对的方法知识已经整理到当前文档，历史原件继续保留在本地。
