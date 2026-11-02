# 外部模型与数据

源码仓库不分发数据集、业务图像、视频、记忆库或预训练权重。`artifacts/models/backbones/manifest.json` 是现有加载器使用的权重规格，保留了其完整性校验。

## 骨干权重

使用 [torchvision 0.21 ResNet 官方实现](https://docs.pytorch.org/vision/0.21/_modules/torchvision/models/resnet.html) 中的 ImageNet-1K V2 权重：

- [Wide ResNet-50-2 权重](https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth)
- [ResNet-50 权重](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)

`python scripts/download_backbones.py --destination <权重目录>` 下载默认骨干并执行已有大小和 SHA-256 校验。需要离线使用时，在有权访问资源的环境准备同名文件和清单，再配置 `ADAPTIVEAOI_BACKBONE_ROOT`。加载期间不自动联网。

[torchvision 的模型许可说明](https://github.com/pytorch/vision#pre-trained-model-license) 指出，预训练模型可能附带训练数据衍生的使用条件。软件包许可证不能代替模型或数据的授权。

## 自有图像

目标域页面使用正常文件夹及可选缺陷文件夹。正常、校准、独立验证和检测输入应分别保存，不要把检测输出、测试目录或其他资源目录当作正常样本来源。可使用有授权的业务图像，或先运行合成示例检查流程。

## 公开数据适配

`ADAPTIVEAOI_DATA_ROOT` 指向下列父目录，名称由适配器识别：

```text
data-root/
├── mvtec/                    # 或 mvtec_ad/
│   └── bottle/
│       ├── train/good/
│       ├── test/good/
│       ├── test/<defect>/
│       └── ground_truth/<defect>/
├── visa/                     # 或 VisA/
│   ├── split_csv/1cls.csv
│   └── <category>/Data/Images/...
└── dagm/                     # 或 DAGM/
    └── Class1/
        ├── Train/
        └── Test/
```

VisA 还支持 `visa_pytorch` 布局，细节见 `datasets/manifest.py`；DAGM 适配器会输出结构检查报告。图像路径、类别、分割和掩码由 `scripts/build_manifest.py` 生成到 manifest 中；检查内容后再构建记忆库。

数据应从各自提供方取得并遵守其条件：

- [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)：页面说明采用 CC BY-NC-SA 4.0，包含非商业用途限制。
- [VisA 提供方项目](https://github.com/amazon-science/spot-diff)：按其数据说明及随数据提供的条款使用。
- DAGM：适配代码不构成数据使用许可，请从有权提供该数据的来源取得授权。

本项目不提供数据镜像，也不把外部数据集上的校准结果作为独立测试结果。
