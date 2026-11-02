# 环境与运行配置

推荐 Python 3.10，依赖版本见根目录的 requirements 文件。运行 `python -m pip check` 可检查当前环境的依赖关系。FAISS 是可选加速项，不是建立正常记忆库的必要条件。

## 安装目录与缓存

在 Windows PowerShell 中可把环境、包缓存和安装中间文件放到 E 盘：

```powershell
$env:PIP_CACHE_DIR = 'E:/codex/download/AdaptiveAOI/pip-cache'
$env:TEMP = 'E:/codex/download/AdaptiveAOI/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
conda create --prefix E:/codex/download/AdaptiveAOI/env python=3.10 -y
conda activate E:/codex/download/AdaptiveAOI/env
python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-test.txt
```

这些设置只影响当前终端及其子进程。CUDA 驱动必须与所选 PyTorch 轮子兼容；本项目没有安装系统驱动的脚本。

## 配置优先级

- `configs/system_integration.yaml`：后端路径、模型条目、默认设备、端口和任务参数。
- `configs/software_config.yaml`：前端 API 地址和默认算法配置。
- `configs/default.yaml`：算法公共默认值；其他算法 YAML 会覆盖相应字段。
- `configs/path_config.yaml`：通用命令行工具的数据根目录。
- `.env`：由后端配置加载器读取，已有进程变量优先。请勿提交本机 `.env`。

`ADAPTIVEAOI_BACKBONE_ROOT` 指向含 `manifest.json` 与权重的目录；`ADAPTIVEAOI_DATA_ROOT` 指向公开数据适配器使用的数据总目录。它们不必位于项目内部。界面的自有样本文件夹单独选择。

通用 CLI 不自动加载 `.env`，运行前在终端设置上述变量。一键启动的端口和 Python 选择也读取当前终端变量；`ADAPTIVEAOI_PYTHON` 应为解释器完整路径。手动分开启动时，用 `ADAPTIVEAOI_INTEGRATION_API_URL` 指定前端连接地址。

## 检查

```powershell
python -c "import torch, torchvision, fastapi, streamlit, cv2; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python scripts/check_video_runtime.py
python -m pytest -q -p no:cacheprovider
```

已有测试大多使用合成输入或隔离的临时目录，不要求真实数据集。完整检测需要另行提供骨干权重，并构建正常记忆库。启动后 API 可先存活，再在后台准备检测资源；不可把 HTTP 200 等同于算法已就绪。
