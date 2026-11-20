# 道路资产检测与追踪系统

基于YOLO26l-737B进行道路资产检测，结合MoCo外观特征、相机运动补偿与离线全局关联生成资产轨迹。

## 系统组成

- **检测**：YOLO26l，17类道路资产，输入尺度960。
- **外观特征**：MoCo v3 ResNet50，2048维归一化特征，按需提取整框和局部特征。
- **追踪**：分类NMS、几何候选、全局关联及组合标牌身份处理。
- **训练数据**：737张审核原图及画质处理版本，3,447个标注框。
- **测试数据**：121张原图，588个标注框。

## 目录

| 路径 | 内容 |
|---|---|
| run.py | 视频检测与追踪 |
| train.py | YOLO训练 |
| evaluate.py | 测试集评估 |
| prepare_videoquality.py | 训练图像画质处理 |
| verify_dataset.py | 图片及标注校验 |
| runtime/ | 检测、特征与关联模块 |
| models/ | 737B、MoCo和通用预训练权重 |
| config/ | 类别、网络结构与参数 |
| data/road-assets737/ | 训练、测试图像与YOLO标注 |
| training/ | 实际训练参数、画质配方与100轮日志 |
| docs/ | 网络、操作和训练文档 |

## 获取与安装

```bash
git lfs install
git clone https://github.com/L1phi/road-asset-tracking-delivery.git
cd road-asset-tracking-delivery
git lfs pull
```

运行环境：Linux、Python3.11、CUDA；GPU配置支持H20与4090系列。使用独立Python环境安装：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

Linux需安装OpenCV所需系统库，Debian/Ubuntu通常为`libgl1`和`libglib2.0-0`。

## 视频检测与追踪

输入为目录顶层的1920×1080、10FPS AVI视频，输出目录使用新路径：

```bash
python run.py --verify-package
python run.py --check-only --input /data/videos --output /data/job-001 --profile h20
python run.py --input /data/videos --output /data/job-001 --profile h20
```

4090系列使用`--profile 4090`。默认YOLO batch=1、MoCo batch=32。程序逐帧处理，不自动修改帧率或分辨率。

输出包括`results/*_tracks.json`、`execution.log`和本次运行参数。JSON中的box为原图xyxy像素坐标，ID按每视频、每类别独立编号。当前不跨视频合并身份；护栏、绿化不参与追踪，路灯采用几何关联。

## 训练与评估

数据和模型已随库提供，可直接检查和训练：

```bash
python verify_dataset.py
python train.py --output outputs/retrain-737b --check-only
python train.py --output outputs/retrain-737b
python evaluate.py --output outputs/test121-evaluation
```

训练从通用`models/yolo26l.pt`初始化，100轮、batch9、imgsz960、seed42，输出使用最后一轮`last.pt`。`models/last.pt`是已训练的737B检测权重。

画质训练图已经提供；`prepare_videoquality.py`用于按记录配方生成新的画质数据。现有画质图为配方重建版本，具体处理和哈希记录见训练文档。

## 技术文档

- [网络结构与算法](docs/网络结构与算法.md)：逐层结构、MoCo预处理、NMS与关联参数。
- [工作流与操作](docs/工作流与操作.md)：处理顺序、输入输出和故障排查。
- [训练与数据集](docs/训练与数据集.md)：数据划分、画质处理、实际训练配置和评分方法。

静态检测test121结果：Micro P=83.31%、R=89.12%、F1=86.11%；固定confidence≥.359、同类别IoU≥.5。完整指标见`docs/test121_metrics.json`。视频轨迹数量与跨视频道路库存采用不同统计口径。

## 依赖与许可

Ultralytics采用AGPL-3.0/商业许可，企业部署按实际使用方式选择授权；MoCo、ResNet和其他依赖遵循各自许可证。道路数据的使用与再分发按双方数据授权约定处理。
