---
kind: configuration_system
name: 基于命令行参数与 JSON 索引的轻量配置系统
category: configuration_system
scope:
    - '**'
source_files:
    - environment_variables/environment_variables/ctde_ppo_baseline_train.py
    - environment_variables/environment_variables/信息转换.py
    - environment_variables/environment_variables/dataset/dataset_index.json
---

本仓库未引入专用配置框架（如 Hydra、OmegaConf、Pydantic Settings），而是采用“默认字典 + argparse 覆盖 + JSON 数据索引”的轻量组合方式，将训练超参、运行开关与数据集元信息分别管理。

1. 运行时配置来源与优先级
- 硬编码默认值：DEFAULT_TRAIN_CONFIG（ctde_ppo_baseline_train.py）集中声明所有可配置键及默认值，是配置的权威基线。
- 命令行覆盖：main() 中通过 argparse 定义一组 --xxx 参数，仅对需要频繁切换的超参暴露入口；解析后以键值形式合并到 config dict。
- 归一化与校验：normalize_training_config(config) 负责类型转换、范围裁剪、枚举校验（如 observation_profile、reward_profile、lr_adapt_mode）、别名映射（use_scene_uav_params → use_metadata_uav_params、init_percentile ↔ init_area_percent），并补齐派生字段（如 observation_profile_dims）。
- 输出固化：训练启动时把最终生效的 config 写入 outputs/<run>/config.json，保证实验可复现。

2. 数据集与场景配置
- 数据集清单由 environment_variables/dataset/dataset_index.json 维护，包含 schema、splits（train/validation/generalization/stress）、raster_files 映射以及每个 scene 的路径/难度/风场等元信息。
- 信息转换.py 中的 DatasetIndex 类负责加载该 JSON、按 split 返回 scene_key 列表、解析相对路径为绝对路径、校验必需文件存在性，并提供 required_file_paths 预检。
- 训练脚本通过 config["data_dir"] 指定 dataset 根目录，再经 DatasetIndex(data_dir) 读取索引，从而解耦“代码路径”和“数据路径”。

3. 架构与约定
- 单一职责：训练脚本只关心“超参与运行开关”，数据位置与场景清单交给 JSON 索引；两者在 ctde_ppo_baseline_train.py 中通过 config["data_dir"] 耦合。
- 分层组织：map/{Train,Validation,Generalization,Stress}/{area}/sceneN/ 下每个场景自带 metadata.json，配合顶层 dataset_index.json 形成“全局索引 + 局部元数据”的双层结构。
- 结果隔离：每次运行生成独立时间戳子目录，内部再分 训练结果/ 与 figures/，其中 训练结果/<exp>/config.json 记录本次运行的完整配置快照。

4. 开发者应遵循的规则
- 新增超参：先在 DEFAULT_TRAIN_CONFIG 中添加默认值，再在 normalize_training_config 中补充类型/范围/枚举校验，最后按需暴露 --xxx 命令行参数。
- 不要直接修改 dataset_index.json 以外的数据清单；如需新增场景，更新该 JSON 的 splits 与 scenes 条目，保持 schema 兼容。
- 路径一律使用相对路径或 source_root 相对路径，由 DatasetIndex 统一解析为绝对路径，避免在不同工作目录下失效。
- 若需新增环境变量注入，应在 normalize_training_config 中显式处理并加入校验，不要散落在业务逻辑里读 os.environ。
- 任何影响模型行为的配置变更都应体现在输出的 config.json 中，以便事后审计与复现实验。