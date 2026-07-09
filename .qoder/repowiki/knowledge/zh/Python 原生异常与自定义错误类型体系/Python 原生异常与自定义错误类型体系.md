---
kind: error_handling
name: Python 原生异常与自定义错误类型体系
category: error_handling
scope:
    - '**'
source_files:
    - environment_variables/environment_variables/ctde_ppo_baseline_train.py
    - environment_variables/environment_variables/信息转换.py
    - environment_variables/environment_variables/test_fire_scene_data.py
---

本仓库采用 Python 原生异常机制进行错误处理，未引入第三方错误库或统一错误码框架。整体风格以“尽早失败 + 语义化异常类型”为主，辅以少量 try/except 包裹外部 I/O 调用并包装为 RuntimeError。

1. 系统/方法概述
- 参数校验：在配置归一化与 Agent 构造处大量使用 raise ValueError(...)，对枚举值、取值范围做显式断言（如 observation_profile、reward_profile、lr_adapt_mode、init_area_percent、init_percentile）。
- 资源缺失：文件/目录不存在时抛出 FileNotFoundError，并附带上下文路径信息（dataset_index.json、metadata.json、scene directory、static_map、raster 等）。
- 数据一致性：栅格形状不匹配、热场计算前置条件不满足时抛出 RuntimeError，消息中包含场景 key 与具体字段名，便于定位。
- 领域错误：定义自定义异常类 InvalidSceneError(RuntimeError)，用于表达“场景无法提供有效 t=0 火边界”这类业务级不可恢复状态。
- 外部依赖 I/O：rasterio/scipy 读取栅格、ASC 文件时，用 try/except Exception as exc 捕获底层异常并重新抛出 RuntimeError(f"Failed to read ...") from exc，保留原始 traceback。
- 日志兜底：通过 TeeStream 将 stdout/stderr 双写到 train_console_log.txt，确保训练过程中的异常堆栈也能持久化到输出目录。

2. 关键文件与位置
- environment_variables/environment_variables/ctde_ppo_baseline_train.py
  - normalize_training_config / CTDE_PPO_Agent.__init__ 中集中进行参数合法性检查，抛出 ValueError。
  - 训练循环中使用 try/except 包裹外部子进程与文件操作，必要时 raise RuntimeError。
- environment_variables/environment_variables/信息转换.py
  - 定义 InvalidSceneError(RuntimeError) 领域异常。
  - DatasetIndex/FireSceneData 中对 dataset_index.json、metadata.json、scene_dir、static_map、核心栅格缺失均抛 FileNotFoundError；对栅格形状不一致、wind field 维度不匹配抛 RuntimeError。
  - load_raster/load_asc 使用 try/except 捕获底层异常并包装为 RuntimeError。
- environment_variables/environment_variables/test_fire_scene_data.py
  - 通过 unittest 断言异常类型与消息片段（assertRaisesRegex），验证错误传播行为。

3. 架构与约定
- 分层策略：
  - 顶层脚本负责用户输入与配置校验（ValueError）；
  - 数据加载层负责资源存在性与格式一致性（FileNotFoundError/RuntimeError）；
  - 业务层对“无效场景”等语义错误使用自定义 InvalidSceneError，上层可据此跳过坏样本而非崩溃。
- 无全局错误码/统一基类：除 InvalidSceneError 外，未定义统一的 BaseError 或错误码枚举，错误类型选择遵循 Python 标准库惯例。
- 无中间件/装饰器模式：错误处理直接内联在函数开头或 I/O 调用周围，没有跨模块的 error middleware。

4. 开发者应遵循的规则
- 参数校验优先：在函数入口对枚举、范围、必填项做显式检查，使用 ValueError 并给出可选值列表或范围说明。
- 资源缺失明确化：找不到文件或目录时抛 FileNotFoundError，并在消息中包含实际路径与当前工作目录，方便复现。
- 数据一致性用 RuntimeError：形状不匹配、字段缺失、计算前置条件不满足时使用 RuntimeError，并带上 scene_key 与字段名。
- 外部 I/O 包装：对 rasterio/scipy/os 等可能失败的调用使用 try/except Exception，重新抛出 RuntimeError 并保留原始异常链（from exc）。
- 领域错误自定义：当错误反映业务语义（如“t=0 火边界为空”）时，继承 RuntimeError 定义专用异常类（参考 InvalidSceneError），以便上层针对性处理。
- 避免吞掉异常：测试用例已覆盖主要异常分支，新增逻辑应补充相应 assertRaises 用例，保证错误路径可回归。