# 第三方软件与资源说明

主项目没有可确认的许可证，本说明不授予主项目或外部资源的使用权，也不替代原许可证。

本目录通过依赖管理文件引用第三方包，未复制这些包的源码、二进制库、数据集和预训练权重。安装时应保留包内的版权、许可证、NOTICE 与作者署名；重新分发运行环境时需按实际包及其传递依赖另行整理。

| 组件 | 用途 | 官方来源 |
|---|---|---|
| PyTorch / torchvision | 张量运算、预训练网络、特征提取 | [PyTorch](https://github.com/pytorch/pytorch)、[torchvision](https://github.com/pytorch/vision) |
| Streamlit / Altair | 页面和可视化 | [Streamlit](https://github.com/streamlit/streamlit)、[Altair](https://github.com/vega/altair) |
| FastAPI / Starlette / Pydantic / Uvicorn | API、校验和服务 | [FastAPI](https://github.com/fastapi/fastapi)、[Starlette](https://github.com/Kludex/starlette)、[Pydantic](https://github.com/pydantic/pydantic)、[Uvicorn](https://github.com/Kludex/uvicorn) |
| NumPy / SciPy / scikit-learn / pandas | 数值、检索、表格与统计 | [NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[scikit-learn](https://scikit-learn.org/)、[pandas](https://pandas.pydata.org/) |
| OpenCV / Pillow | 图像与视频处理 | [OpenCV Python](https://github.com/opencv/opencv-python)、[Pillow](https://github.com/python-pillow/Pillow) |
| FAISS | 可选近邻检索 | [FAISS](https://github.com/facebookresearch/faiss) |
| streamlit-drawable-canvas-fix | 反馈掩码绘制 | [PyPI 包说明](https://pypi.org/project/streamlit-drawable-canvas-fix/) |

`streamlit-drawable-canvas-fix` 的已安装元数据没有顶层 License 字段，需要以实际发行包携带的许可文件为准。仓库只声明包依赖，未将其前端组件或依赖副本纳入发布内容。

PatchCore 方法参考 [Karsten Roth 等人的原论文](https://arxiv.org/abs/2106.08265)。方法引用不代表复制了其官方实现，也不替代代码归属核查。若后续发现复制的第三方片段，应保留原始署名并补充其适用许可。

模型与数据的来源及条件见 [资源说明](docs/resources.md)。本地绘制的应用图标使用 `launchers/generate_icon.py` 中的几何图形和 AOI 文本生成，不包含外部标志。
