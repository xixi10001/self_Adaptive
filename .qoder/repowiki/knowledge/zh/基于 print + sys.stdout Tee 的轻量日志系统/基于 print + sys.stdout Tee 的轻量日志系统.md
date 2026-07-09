---
kind: logging_system
name: 基于 print + sys.stdout Tee 的轻量日志系统
category: logging_system
scope:
    - '**'
source_files:
    - environment_variables/environment_variables/ctde_ppo_baseline_train.py
    - environment_variables/environment_variables/test_console_log_text.py
---

本仓库未引入任何第三方日志框架（如 logging、loguru、structlog），而是采用“print + sys.stdout/stderr Tee”的极简方案，将训练过程的文本输出同时写入控制台与本地文件。核心约定如下：

1. 输出机制
- 通过 `ctde_ppo_baseline_train.py` 中的 `TeeStream` 类与 `setup_console_tee(log_path)` 函数，在进程启动时把 `sys.stdout` / `sys.stderr` 替换为双写对象，所有后续 `print(...)` 都会同步追加到指定 log 文件。
- 该 Tee 支持多实例切换（相同路径复用，不同路径先关闭旧实例再重建），并以 UTF-8 编码、行缓冲模式打开文件。
- 训练脚本入口会调用 `setup_console_tee(os.path.join(output_dir, CONSOLE_LOG_NAME))`，其中 `CONSOLE_LOG_NAME = "train_console_log.txt"`，每个实验子目录均生成一份独立的控制台日志。

2. 结构化记录
- 除控制台文本外，训练循环维护一个大型 Python dict `training_log`，按列收集 episode/ppo_update 级别的指标（rewards、task_scores、approx_kl、actor_lr 等），并在训练结束时序列化为 JSON 文件；验证集指标则保存在 `validation_log` 中。
- 实验配置统一以 `config.json` 形式写入输出目录，便于复现实验。

3. 日志级别与内容约定
- 无显式 level 概念，关键阶段事件（课程阶段切换、学习率调整、图表生成提示）使用带前缀的中文 `print` 语句，例如：
  - `课程阶段 X -> Y | 本阶段回合=... | 成功率=...% | 覆盖率=...%`
  - `[curriculum] env.init_area_percent -> ...`
  - `[stage3 curriculum] target ... -> ...`
  - `[near curriculum] near_prob ... -> ...`
- 单元测试 `test_console_log_text.py` 断言这些中文标签出现在 stdout 中，从而保证日志文案不被随意修改。

4. 输出目录结构
- 每次运行在 `outputs/<时间戳>/<实验名>/` 下生成：
  - `config.json`：归一化后的训练配置
  - `train_console_log.txt`：完整控制台日志（含 print 输出）
  - `logs/`：由绘图脚本生成的中间日志或图表数据
  - `figures/`、`summary_curves/`：训练/泛化曲线图
  - `训练结果/`：模型权重与训练摘要

5. 开发者应遵循的规则
- 新增诊断信息优先使用 `print(...)` 并通过 `setup_console_tee` 自动落盘，不要直接操作 `open(...)` 写日志文件。
- 保持现有中文标签不变，以便测试用例继续通过；如需新增阶段，请在对应位置添加可被断言的固定字符串。
- 数值型指标仍应写入 `training_log` / `validation_log` dict，最终序列化到 JSON，避免仅依赖文本解析。
- 不要在业务模块内自行 `import logging` 或使用其他日志库，以免破坏统一的 Tee 重定向行为。