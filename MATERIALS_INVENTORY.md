# 代码材料清单与审查入口

本包对应 Receiver-State Hypothesis Selection for Doppler-Only LEO Positioning under Residual Blindness。当前用途是作者私有审查，包含代码、冻结配置和说明。没有原始数据、处理后的数据、论文图表、稿件、ESM、内部审稿报告或未完成的临时副本。本次整理没有执行实验。

## 文件作用

| 材料 | 作用 | 阅读或使用方式 |
|---|---|---|
| [CODE_FILE_CATALOG.md](CODE_FILE_CATALOG.md) | 逐文件用途索引；列出源码说明、主要函数或类、配置用途及来源角色 | 按完整相对路径定位；函数列表来自源码静态解析，不代表所有函数都应直接运行 |
| [DATA_SOURCES.md](DATA_SOURCES.md) | 原始数据提供方链接、所选记录、放置位置和历史输入身份 | 只从原提供方下载，按其使用条件处理 |
| [REPRODUCTION.md](REPRODUCTION.md) | 环境准备、各论文部分的手动执行顺序、阶段输入和限制 | 按所属实验的冻结运行时执行，避免混用快照 |
| [src/leo_positioning/](src/leo_positioning/) | 测量模型、M0–M14候选模型、求解器、投影和诊断计算 | 核心科学模块；部分旧版本是冻结运行时的依赖，不代表新增论文方法 |
| [src/flagship/native_runtime.py](src/flagship/native_runtime.py) | 冻结候选运行时的工程接口 | 候选拟合与最终选择应分别确认，不能把拟合完成等同于选择流程完成 |
| [src/flagship/selector_replay.py](src/flagship/selector_replay.py) | v0.5.2选择规范解释与重放 | 使用冻结JSON/CSV规则；输入是阶段生成的候选诊断记录，不直接输入原始射频数据 |
| [src/flagship/information.py](src/flagship/information.py)、[geometry_kernels.py](src/flagship/geometry_kernels.py) | 信息矩阵及几何计算 | 保持相同观测、权重、单位和干扰状态定义 |
| [src/flagship/controlled_motion_clock.py](src/flagship/controlled_motion_clock.py) | 受控运动与时钟观测生成代码 | 结合冻结配置使用；不包含生成后的观测或结果 |
| [src/flagship/d2_records.py](src/flagship/d2_records.py) | D2记录与报告时刻处理 | 最终报告时刻为110 s，完整观测窗口为0–120 s |
| [src/flagship/orbits.py](src/flagship/orbits.py) | 轨道输入和传播接口 | 历史D2必须使用对应快照，当前轨道下载不能替代历史身份 |
| [src/flagship/extract_gnss.py](src/flagship/extract_gnss.py)、[gps_l1_wrapper.py](src/flagship/gps_l1_wrapper.py) | GVINS原始GNSS提取和GPS-L1物理接口 | 支持GPS MEO机制分析；不代表动态LEO真实射频性能基准 |
| [manual_workflows/tree/](manual_workflows/tree/) | measured-static、development、D2、UrbanNav、GVINS的历史分阶段脚本与冻结依赖 | 目录中的 `tmp/d2_execution_r1` 是已核验的历史源码位置，与被排除的未完成临时副本不同 |
| [configs/](configs/) | 候选门控、选择排序、回退、实验参数和数据格式配置 | CSV属于规则配置时予以保留；没有实验结果CSV |
| [environment/](environment/) | 历史Python环境和依赖版本记录 | 建议在独立环境中安装，版本检查不等于数值复现 |
| [third_party/](third_party/) | 原项目使用的ROS消息定义及其许可证 | 保留来源与许可说明；公开前另行检查派生代码许可 |
| [check_code.py](check_code.py) | 文件哈希、语法、包内依赖和可选导入检查 | `python -B check_code.py --imports`，不执行实验 |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | 原源码出处与复制身份 | 用于区分实验冻结快照；没有按名称合并不同版本 |
| [MANIFEST.json](MANIFEST.json)、[SHA256SUMS.txt](SHA256SUMS.txt) | 当前交付文件清单和完整性校验 | 校验下载后的文件；Git内部目录不属于代码交付清单 |
| [LICENSE_NOTICE.md](LICENSE_NOTICE.md) | 作者许可待确认事项及第三方许可边界 | 私有审查期间也应保留第三方许可证 |

## 原始数据链接

| 来源 | 官方入口 | 本文使用范围 |
|---|---|---|
| Iridium实测Doppler | [Certifiable-Doppler-positioning](https://github.com/Baoshan-Song/Certifiable-Doppler-positioning) | 实测静态案例及受控观测所用几何 |
| UrbanNav | [UrbanNavDataset](https://github.com/IPNL-POLYU/UrbanNavDataset) | Tokyo 20181219的Odaiba与Shinjuku运动记录 |
| GVINS | [GVINS-Dataset](https://github.com/HKUST-Aerial-Robotics/GVINS-Dataset) | sports_field原始GNSS及广播星历 |
| CelesTrak | [GP/OMM官方说明](https://celestrak.org/NORAD/documentation/gp-data-formats.php) | 轨道元素；D2历史快照的日期和SHA见DATA_SOURCES |
| Iridium误差源资料 | [Mendeley登记记录v2](https://doi.org/10.17632/xcxspv8c2r.2) | 精确历史子集的对应关系仍需作者核对，不能当作已经确认的逐文件替代来源 |

2026-09-15再次读取了前四个官方入口。Mendeley DOI在本轮访问工具中未成功返回页面，因此仅保留原有登记链接，不新增“本轮已验证可下载”的承诺。下载文件和历史快照不随仓库提供。

## 复现流程简介

1. 下载本仓库到较短的本地路径，阅读许可证及环境记录，建立独立Python环境。
2. 根据DATA_SOURCES下载相应原始数据，核对版本、日期、文件名和已登记哈希，配置本地输入路径。
3. 按REPRODUCTION选择需要复现的论文部分，运行对应数据准备阶段，生成中间观测、清单和初始化记录。
4. 用该部分自己的冻结运行时拟合候选，再按相应选择规范处理候选诊断；保持种子、阈值、观测范围、报告时刻和比较基准不变。
5. 运行对应评估阶段，并按论文与ESM的实验身份核对结果。绘图脚本属于源码，已生成的图表不在仓库中。

本包提供手动流程简介，不提供一键复现保证。只有代码和原始公开入口，仍可能不足以重建逐字节相同的历史输入清单；需要精确历史快照、初始化记录或阶段身份文件时，请向通讯作者申请。已完成的包内哈希、语法和导入检查不等同于从原始数据全流程复现实验。
