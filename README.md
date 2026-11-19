# Road Asset Tracking — 道路资产检测与追踪系统

验证范围：已完成文件完整性、模型身份、代码语法及入口基础测试；企业目标GPU端到端验证按验收清单执行。文件完整性不等同于准确率或生产验收。

本仓库只交付737B检测、MoCo及冻结全局图V2链路。研究档案保存在另一个私有仓库`L1phi/road-asset-tracking`，不属于本仓库或企业部署文件。模型通过Git LFS提供；访问需要仓库授权。

## 包含与不包含

包含：推理入口`run.py`、训练入口`train.py`、画质准备`prepare_videoquality.py`、评估入口`evaluate.py`、冻结运行模块、模型权重、737训练/121测试及审核GT、配置、历史训练参数与100轮日志、校验和操作文档。

不包含：Qwen/LocateAnything/YOLO11/737A旧实验、原始道路视频、提示词与个人清理记录、测速全量归档、运行环境副本、旧可视化。此处仅增加与737B直接相关的审核数据和训练证据，不把整个研究目录重新混入交付。

## 完整文档导航

| 要了解的问题 | 文档/入口 |
|---|---|
| 网络有哪些模块、每层如何连接、MoCo如何提特征 | [网络结构与算法](docs/网络结构与算法.md) |
| 处理的先后顺序、如何部署和操作、怎么读结果 | [工作流与操作](docs/工作流与操作.md) |
| 数据从哪里来、如何降质、怎么训练和评估 | [训练与数据集](docs/训练与数据集.md) |
| 17个类别ID与名称 | [固定类别表](config/classes.json) |
| 官方YOLO26逐层结构 | [固定版本结构配置](config/yolo26_official_v8.4.146.yaml) |
| 当年实际训练参数、100轮日志、画质配方 | training/args_original.yaml、training/results_original.csv、training/experiment_record.json |
| 数据数量、图像和GT身份 | data/road-assets737/audit.json、dataset_manifest.json、python verify_dataset.py |
| 验证范围和企业上线检查 | [验收与边界](docs/验收与边界.md)、validation.json |

它不是一个端到端新网络：YOLO负责检测，MoCo负责外观特征，GMC/NMS/候选和图关联是独立算法模块。只有YOLO在道路资产数据上微调；MoCo和规则没有在本项目重新训练。

## 支持范围

- Linux、Python3.11、CUDA GPU；已记录实验硬件为H20与4090 D。
- 输入目录顶层的`.avi`文件，1920×1080、10FPS；逐帧处理，不自动抽帧或改分辨率。
- 单视频离线关联，不跨视频去重；护栏/绿化排除，路灯只做几何关联，桩号连续高置信度确认。
- 输出检测框对应轨迹，不输出已验收道路库存。组合牌、相似目标、误检与漏检仍有已知限制。
- 新入口使用单进程串行基准，**不承诺历史多进程最优配置的吞吐**。长视频的全局关联内存/耗时尚未验收，历史测试主要是10秒短片。

## 获取与安装

```bash
git lfs install
git clone https://github.com/L1phi/road-asset-tracking-delivery.git
cd road-asset-tracking-delivery
git lfs pull
```

在独立Python3.11环境安装，勿装到系统/base：

Linux容器还需要系统动态库`libGL.so.1`与GLib（Debian/Ubuntu包名通常为`libgl1`、`libglib2.0-0`）。不要同时安装多个提供cv2命名空间的OpenCV发行包；缺库时先由企业运维安装系统依赖，不擅自切换冻结版本。

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

依赖中的非核心版本是交付约束，不代表新组合已经做过GPU验收。原测试环境证据见`docs/reference_environment.json`。

## 唯一运行入口

```bash
python run.py --verify-package
python run.py --check-only --input /data/videos --output /data/job-001 --profile h20
python run.py --input /data/videos --output /data/job-001 --profile h20
```

训练与评估入口：

```bash
# 未生成视频画质训练图时先运行；已存在时脚本拒绝覆盖
python prepare_videoquality.py
python train.py --output outputs/retrain-737b --check-only
python train.py --output outputs/retrain-737b
python evaluate.py --output outputs/test121-evaluation
```

训练前须准备指定SHA256的通用预训练模型，不能从已交付道路last.pt冒充从头微调；详见训练文档。本次文档与数据整理没有重新运行100轮训练。

4090系列用`--profile 4090`。输出目录必须不存在；只检查不会创建输出。日志在`execution.log`，轨迹在`results/*_tracks.json`。MoCo加载/实际推理只在正式运行时验证；check-only不等于完整GPU烟测。首版先用默认YOLO batch=1、MoCo batch=32，确认链路后再测性能。

入口只做配置、校验和调度：为当前输入生成manifest，将原样冻结模块复制到输出的`_runtime`目录再运行，不改关联代价、阈值或模型。旧内核的`--all29`参数在本入口中表示当前manifest全部视频，不限固定29段。

## 输出约定

`predictions`按零起始帧号组织；每条包含class_id、box（原图xyxy像素）、confidence、内部id等。`display_map`将内部ID映射为各类独立编号；不同视频的同号不代表同一资产。`class_statistics`的`IDs_not_inventory`不能直接当库存。NMS/关联/大小框审计保留在同一轨迹JSON。原始框与分数不由可视化修饰。

## 必须完成的验收

见`docs/验收与边界.md`。交付方应在企业GPU上检查：依赖、两个模型哈希、MoCo实际推理、已知短片结果一致性、新视频全链路、长视频资源压力、计数误差与日志。没有通过前，不标“生产可用”或承诺实时性能。

## 第三方权利与数据权限

YOLO实现依赖Ultralytics，需核对其AGPL-3.0/商业许可与企业部署方式；MoCo/ResNet及其他依赖遵循各自许可证。没有擅自给第三方权重或代码授予新的统一商业许可证。随库训练/测试图像含真实道路画面，需按双方约定处理数据授权和隐私；Private仓库访问权不等于后续公开再分发许可。推理用新视频由企业另行提供。
