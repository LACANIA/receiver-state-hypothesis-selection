# Receiver State Hypothesis Selection

Receiver-State Hypothesis Selection for Doppler-Only LEO Positioning under Residual Blindness

本目录按照作者最新要求，仅提供源码、必要配置、依赖说明、原始数据链接和简要复现流程。原始数据、处理后的数据、结果表、论文图件、稿件、ESM及内部审查报告均未纳入本包。少量 CSV 是选择器规则配置或源码校验清单，不含实验结果表。

## 从哪里开始

先阅读 [材料清单](MATERIALS_INVENTORY.md)，然后阅读 [原始数据链接](DATA_SOURCES.md) 和 [分阶段复现流程](REPRODUCTION.md)。[逐文件用途索引](CODE_FILE_CATALOG.md) 列出代码、配置与依赖文件的作用和可定位的函数。当前交付采用手动分阶段方式；没有提供一键复现功能，也没有在本次整理中重新运行实验。

| 目录或文件 | 内容 |
|---|---|
| `src/leo_positioning/` | 冻结的测量模型、候选估计器和共用计算函数 |
| `src/flagship/` | 选择器重放、轨道、信息矩阵、D2状态处理、受控运动与GPS-L1接口 |
| `configs/` | 冻结配置、选择规则、数据格式和消息定义校验身份 |
| `manual_workflows/tree/` | 原工程的分阶段源码及相应冻结依赖；保留可辨认的目录关系 |
| `environment/` | 原环境记录和依赖列表 |
| `third_party/` | ROS消息定义及其原许可证 |
| `check_code.py` | 文件哈希、Python语法和导入检查；不进行实验 |
| `SOURCE_MANIFEST.json` | 源码出处和复制前后哈希 |

相同模块可能出现在多个冻结快照中，这是为了保留各实验原有依赖身份。没有合并成一个新版求解器。复制时仅对私人路径及机器环境名称进行了必要调整，科学计算函数与参数不作新方法修改。

## 简单检查

安装依赖后，在本目录执行 `python -B check_code.py --imports`。这项检查只证明文件、语法和指定模块导入通过；不等同于已完成从原始数据到论文结果的数值复现。历史脚本包含阶段输入和哈希检查，运行前应按流程配置本地路径及新生成的输入清单，不应关闭科学身份检查。

## 范围与许可

正式科学范围为 M0–M14、当前EGSHS、measured-static机制案例、development、原D2，以及论文采用的几何、motion-clock、UrbanNav、GVINS机制分析。新Route T及未采用的新方法没有作为正式方法加入。

作者代码许可仍需作者确认，详见 `LICENSE_NOTICE.md`。本包用于作者私有审查；公开发布需要作者另行确认，私有托管不代表已经公开代码或完成许可审查。此前较大的复现资料目录及获准保留的临时副本均未删除，也未装入本次纯代码ZIP。

English summary: This is a source-only, manual-workflow archive. Download original data from the linked owners, configure local paths, and execute the relevant stages described in REPRODUCTION.md. No datasets, processed outputs, figures or manuscripts are bundled. Packaging checks do not constitute a new scientific run.
