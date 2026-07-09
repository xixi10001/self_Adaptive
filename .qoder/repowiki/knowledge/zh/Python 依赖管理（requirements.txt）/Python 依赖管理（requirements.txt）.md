---
kind: dependency_management
name: Python 依赖管理（requirements.txt）
category: dependency_management
scope:
    - '**'
source_files:
    - environment_variables/requirements.txt
---

本仓库采用最简化的 Python 依赖管理方式，仅通过 `environment_variables/requirements.txt` 声明运行时依赖，未使用虚拟环境、锁文件或私有源配置。

- **包管理器**：pip + requirements.txt。所有第三方库版本以 `>=最低版本` 形式声明，无精确锁定。
- **核心依赖**：numpy、rasterio、matplotlib、scipy、opencv-python。
- **可选依赖**：stable-baselines3、torch、tensorboard 被注释保留，按需启用。
- **缺失机制**：仓库中未发现 `pyproject.toml`、`setup.py`、`Pipfile`、`poetry.lock`、`go.mod`、`package.json` 等任何锁文件或构建清单；也未见 `.venv`、`vendor/`、私有 PyPI 源或 `pip.conf` 配置。
- **约定**：新增依赖应直接追加到 `environment_variables/requirements.txt`，并在相关设计文档中同步记录（如 `docs/superpowers/specs/2026-07-06-thermal-field-optimization-design.md` 明确要求在此文件中声明 SciPy 与 OpenCV）。