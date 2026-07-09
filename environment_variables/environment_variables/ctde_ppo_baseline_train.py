"""Clean CTDE-PPO baseline training script.

This script pairs with rl_environment_baseline.py and keeps only the baseline
CTDE-PPO algorithm plus the task training loop.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rl_environment_baseline import FireSearchBaselineEnvironment


_data_module = importlib.import_module("\u4fe1\u606f\u8f6c\u6362")
DatasetIndex = _data_module.DatasetIndex
SceneManager = _data_module.SceneManager
validate_scene_boundaries = _data_module.validate_scene_boundaries
RESULTS_DIR_NAME = "\u8bad\u7ec3\u7ed3\u679c"
SOURCE_DIR_NAME = "\u8bad\u7ec3\u6e90\u7801"
CONSOLE_LOG_NAME = "train_console_log.txt"
THERMAL_HEALTH_LIMITS = {
    "sat_ratio": 0.10,
    "high_ratio": 0.50,
    "zero_grad_in_high_ratio": 0.20,
}


class TeeStream:
    def __init__(self, stream, log_path: str, mode: str = "a"):
        self.stream = stream
        self.log_path = os.path.abspath(log_path)
        self.file = open(self.log_path, mode, encoding="utf-8", errors="replace", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        self.file.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_TEE_STDOUT = None
_TEE_STDERR = None
_TEE_LOG_PATH = None


def setup_console_tee(log_path: str, mode: str = "a") -> None:
    global _TEE_LOG_PATH, _TEE_STDOUT, _TEE_STDERR
    log_path = os.path.abspath(log_path)
    if _TEE_LOG_PATH == log_path:
        return

    for tee in (_TEE_STDOUT, _TEE_STDERR):
        if tee is not None:
            tee.close()
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _TEE_STDOUT = TeeStream(_ORIGINAL_STDOUT, log_path, mode=mode)
    _TEE_STDERR = TeeStream(_ORIGINAL_STDERR, log_path, mode="a")
    sys.stdout = _TEE_STDOUT
    sys.stderr = _TEE_STDERR
    _TEE_LOG_PATH = log_path


DEFAULT_TRAIN_CONFIG = {
    "data_dir": "./dataset",
    "train_split": "train",
    "eval_split": "generalization",
    "train_scene_keys": None,
    "num_drones": 2,
    "vision_radius": 16,
    "max_steps": 600,
    "use_metadata_uav_params": False,
    "observation_profile": "baseline",
    "reward_profile": "boundary_coverage",
    "norm_params_source": "scene_p99.5",
    "init_percentile": 5.0,
    "init_area_percent": 5.0,
    "total_episodes": 2500,
    "max_train_updates": None,
    "actor_lr": 2e-4,
    "critic_lr": 5e-4,
    "lr_adapt_mode": "fixed",
    "target_kl": 0.010,
    "actor_lr_min": 2e-5,
    "actor_lr_max": 4e-4,
    "kl_ema_beta": 0.9,
    "kl_lr_alpha": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "batch_size": 4096,
    "save_interval": 100,
    "log_interval": 10,
    "seed": 42,
    "comparison_seeds": [42, 43, 44],
    "stage2_success_target": 0.15,
    "stage3_success_target": 0.60,
    "stage3_near_prob": 0.25,
    "validation_split": "validation",
    "validation_interval": 100,
    "validation_episodes_per_scene": 5,
    "save_best_by_validation": True,
    "eval_scene_keys": None,
    "eval_episodes_per_scene": 50,
    "eval_stages": [3],
    "eval_seed_stride": 100,
    "eval_after_train": True,
    "final_eval_splits": ["validation", "generalization", "stress"],
    "final_eval_episodes_per_scene": 50,
    "evaluate_best_val_after_train": True,
    "quality_score_threshold": 0.55,
    "quality_window": 50,
    "quality_tail_fraction": 0.2,
    "quality_target_kl": 0.010,
    "plot_after_train": True,
    "figure_window": 100,
    "figure_dpi": 300,
    "output_root_dir": "./outputs",
    "output_subdir": "baseline_ctde_ppo",
}


def normalize_training_config(config: Dict = None) -> Dict:
    normalized = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    config = {} if config is None else copy.deepcopy(config)
    user_keys = set(config.keys())

    def _normalize_key_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item) for item in value]

    def _normalize_str_list(value):
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return [str(item).lower() for item in value]

    for key, value in config.items():
        if isinstance(normalized.get(key), dict) and isinstance(value, dict):
            merged = copy.deepcopy(normalized[key])
            merged.update(value)
            normalized[key] = merged
        else:
            normalized[key] = value

    normalized["num_drones"] = max(1, int(normalized["num_drones"]))
    normalized["vision_radius"] = max(1, int(normalized["vision_radius"]))
    normalized["max_steps"] = max(1, int(normalized["max_steps"]))
    if "use_scene_uav_params" in user_keys and "use_metadata_uav_params" not in user_keys:
        normalized["use_metadata_uav_params"] = bool(normalized["use_scene_uav_params"])
    normalized["use_metadata_uav_params"] = bool(normalized.get("use_metadata_uav_params", False))
    normalized["use_scene_uav_params"] = bool(normalized["use_metadata_uav_params"])
    normalized["observation_profile"] = str(
        normalized.get("observation_profile", "baseline")
    ).lower()
    if normalized["observation_profile"] not in FireSearchBaselineEnvironment.OBSERVATION_PROFILE_DIMS:
        raise ValueError("observation_profile must be one of: " + ", ".join(sorted(FireSearchBaselineEnvironment.OBSERVATION_PROFILE_DIMS)))
    normalized["reward_profile"] = str(
        normalized.get("reward_profile", "boundary_coverage")
    ).lower()
    if normalized["reward_profile"] not in FireSearchBaselineEnvironment.REWARD_PROFILES:
        raise ValueError("reward_profile must be one of: " + ", ".join(sorted(FireSearchBaselineEnvironment.REWARD_PROFILES)))
    normalized["observation_profile_dims"] = copy.deepcopy(
        FireSearchBaselineEnvironment.OBSERVATION_PROFILE_DIMS
    )
    normalized["norm_params_source"] = str(
        normalized.get("norm_params_source", "scene_p99.5")
    )
    if "init_area_percent" in user_keys:
        normalized["init_percentile"] = normalized["init_area_percent"]
    else:
        normalized["init_area_percent"] = normalized.get("init_percentile")
    if normalized.get("init_area_percent") is not None:
        normalized["init_area_percent"] = float(normalized["init_area_percent"])
        if not 0.0 <= normalized["init_area_percent"] <= 100.0:
            raise ValueError("init_area_percent must be between 0 and 100")
    if normalized.get("init_percentile") is not None:
        normalized["init_percentile"] = float(normalized["init_percentile"])
        if not 0.0 <= normalized["init_percentile"] <= 100.0:
            raise ValueError("init_percentile must be between 0 and 100")
    normalized["total_episodes"] = max(1, int(normalized["total_episodes"]))
    normalized["batch_size"] = max(32, int(normalized["batch_size"]))
    normalized["ppo_epochs"] = max(1, int(normalized["ppo_epochs"]))
    normalized["save_interval"] = max(1, int(normalized["save_interval"]))
    normalized["log_interval"] = max(1, int(normalized["log_interval"]))
    normalized["seed"] = int(normalized["seed"])
    normalized["comparison_seeds"] = [
        int(seed) for seed in normalized.get("comparison_seeds", [normalized["seed"]])
    ]
    normalized["actor_lr"] = float(normalized["actor_lr"])
    normalized["critic_lr"] = float(normalized["critic_lr"])
    normalized["lr_adapt_mode"] = str(normalized.get("lr_adapt_mode", "fixed")).lower()
    if normalized["lr_adapt_mode"] not in {"fixed", "kl"}:
        raise ValueError("lr_adapt_mode must be 'fixed' or 'kl'")
    normalized["target_kl"] = max(1e-8, float(normalized["target_kl"]))
    normalized["actor_lr_min"] = max(1e-12, float(normalized["actor_lr_min"]))
    normalized["actor_lr_max"] = max(normalized["actor_lr_min"], float(normalized["actor_lr_max"]))
    normalized["kl_ema_beta"] = float(np.clip(normalized["kl_ema_beta"], 0.0, 0.999))
    normalized["kl_lr_alpha"] = max(0.0, float(normalized["kl_lr_alpha"]))
    normalized["gamma"] = float(normalized["gamma"])
    normalized["gae_lambda"] = float(normalized["gae_lambda"])
    normalized["clip_epsilon"] = float(normalized["clip_epsilon"])
    normalized["entropy_coef"] = float(normalized["entropy_coef"])
    normalized["value_coef"] = float(normalized["value_coef"])
    normalized["max_grad_norm"] = float(normalized["max_grad_norm"])
    normalized["stage2_success_target"] = float(np.clip(normalized["stage2_success_target"], 0.0, 1.0))
    normalized["stage3_success_target"] = float(np.clip(normalized["stage3_success_target"], 0.0, 1.0))
    normalized["stage3_near_prob"] = float(np.clip(normalized["stage3_near_prob"], 0.0, 1.0))
    normalized["train_split"] = str(normalized.get("train_split", "train")).lower()
    normalized["eval_split"] = str(normalized.get("eval_split", "generalization")).lower()
    normalized["validation_split"] = str(normalized.get("validation_split", "validation")).lower()
    normalized["validation_interval"] = max(1, int(normalized["validation_interval"]))
    normalized["validation_episodes_per_scene"] = max(1, int(normalized["validation_episodes_per_scene"]))
    normalized["save_best_by_validation"] = bool(normalized["save_best_by_validation"])
    normalized["train_scene_keys"] = _normalize_key_list(normalized.get("train_scene_keys"))
    normalized["eval_scene_keys"] = _normalize_key_list(normalized.get("eval_scene_keys"))
    normalized["eval_episodes_per_scene"] = max(1, int(normalized["eval_episodes_per_scene"]))
    normalized["eval_stages"] = [int(stage) for stage in normalized.get("eval_stages", [3])]
    normalized["eval_seed_stride"] = max(1, int(normalized["eval_seed_stride"]))
    normalized["eval_after_train"] = bool(normalized["eval_after_train"])
    normalized["final_eval_splits"] = _normalize_str_list(normalized.get("final_eval_splits", ["validation", "generalization", "stress"]))
    normalized["final_eval_episodes_per_scene"] = max(1, int(normalized["final_eval_episodes_per_scene"]))
    normalized["evaluate_best_val_after_train"] = bool(normalized["evaluate_best_val_after_train"])
    normalized["quality_score_threshold"] = float(np.clip(normalized["quality_score_threshold"], 0.0, 1.0))
    normalized["quality_window"] = max(1, int(normalized["quality_window"]))
    normalized["quality_tail_fraction"] = float(np.clip(normalized["quality_tail_fraction"], 0.05, 1.0))
    if "quality_target_kl" not in user_keys:
        normalized["quality_target_kl"] = normalized["target_kl"]
    normalized["quality_target_kl"] = max(1e-8, float(normalized["quality_target_kl"]))
    normalized["plot_after_train"] = bool(normalized["plot_after_train"])
    normalized["figure_window"] = max(1, int(normalized["figure_window"]))
    normalized["figure_dpi"] = max(72, int(normalized["figure_dpi"]))

    max_train_updates = normalized.get("max_train_updates")
    if max_train_updates is None:
        normalized["max_train_updates"] = None
    else:
        max_train_updates = int(max_train_updates)
        normalized["max_train_updates"] = max_train_updates if max_train_updates > 0 else None

    return normalized


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _task_score(coverage: float, success: bool, length: int, max_steps: int) -> float:
    efficiency = float(success) * (1.0 - np.clip(float(length) / max(float(max_steps), 1.0), 0.0, 1.0))
    return float(0.5 * float(coverage) + 0.3 * float(success) + 0.2 * efficiency)


def _validation_model_score(summary: Dict) -> float:
    task_score = float(summary.get("mean_task_score", 0.0))
    coverage = float(summary.get("mean_coverage", 0.0))
    timeout_rate = float(summary.get("timeout_rate", 0.0))
    zero_timeout_rate = float(summary.get("zero_coverage_timeout_rate", 0.0))
    return float(task_score + 0.10 * coverage - 0.20 * timeout_rate - 0.45 * zero_timeout_rate)


def _rolling_mean(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    window = max(1, min(int(window), arr.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def _first_threshold_crossing(values: List[float], x_values: List[float], threshold: float, window: int):
    arr = np.asarray(values, dtype=np.float64)
    xs = np.asarray(x_values, dtype=np.float64)
    if arr.size == 0 or xs.size == 0:
        return None

    window = max(1, min(int(window), arr.size))
    rolling = _rolling_mean(arr.tolist(), window)
    hit_indices = np.flatnonzero(rolling >= float(threshold))
    if hit_indices.size == 0:
        return None

    original_idx = int(hit_indices[0] + window - 1)
    original_idx = min(original_idx, xs.size - 1)
    return float(xs[original_idx])


def _tail(values: List[float], fraction: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    n_tail = max(1, int(np.ceil(arr.size * float(fraction))))
    return arr[-n_tail:]


def _unique_update_values(training_log: Dict, value_key: str) -> np.ndarray:
    updates = training_log.get("ppo_updates", [])
    values = training_log.get(value_key, [])
    seen = set()
    unique_values = []

    for update_id, value in zip(updates, values):
        update_id = int(update_id)
        if update_id <= 0 or update_id in seen:
            continue
        seen.add(update_id)
        unique_values.append(float(value))

    return np.asarray(unique_values, dtype=np.float64)


def compute_model_quality_metrics(training_log: Dict, config: Dict) -> Dict:
    task_scores = np.asarray(training_log.get("task_scores", []), dtype=np.float64)
    rewards = np.asarray(training_log.get("rewards", []), dtype=np.float64)
    total_steps = np.asarray(training_log.get("total_steps", []), dtype=np.float64)
    ppo_updates = np.asarray(training_log.get("ppo_updates", []), dtype=np.float64)

    window = int(config["quality_window"])
    threshold = float(config["quality_score_threshold"])
    tail_fraction = float(config["quality_tail_fraction"])
    target_kl = float(config["quality_target_kl"])

    metrics = {
        "settings": {
            "score_threshold": threshold,
            "window": window,
            "tail_fraction": tail_fraction,
            "target_kl": target_kl,
        },
        "convergence_efficiency": {},
        "reward_stability": {},
        "kl_stability": {},
    }

    if task_scores.size > 0:
        if total_steps.size == task_scores.size and total_steps[-1] > total_steps[0]:
            auc_task = np.trapz(task_scores, total_steps) / max(total_steps[-1] - total_steps[0], 1.0)
        else:
            auc_task = float(np.mean(task_scores))

        metrics["convergence_efficiency"] = {
            "auc_task_score_by_steps": float(auc_task),
            "steps_to_threshold": _first_threshold_crossing(
                task_scores.tolist(), total_steps.tolist(), threshold, window
            ),
            "updates_to_threshold": _first_threshold_crossing(
                task_scores.tolist(), ppo_updates.tolist(), threshold, window
            ),
        }

        rolling_scores = _rolling_mean(task_scores.tolist(), window)
        if rolling_scores.size >= 2:
            drops = np.maximum(0.0, rolling_scores[:-1] - rolling_scores[1:])
            mean_drop = float(np.mean(drops))
            max_drop = float(np.max(drops))
        else:
            mean_drop = 0.0
            max_drop = 0.0

        tail_scores = _tail(task_scores.tolist(), tail_fraction)
        tail_rewards = _tail(rewards.tolist(), tail_fraction)
        metrics["reward_stability"] = {
            "reward_std_tail": float(np.std(tail_rewards)) if tail_rewards.size > 0 else None,
            "task_score_std_tail": float(np.std(tail_scores)) if tail_scores.size > 0 else None,
            "mean_performance_drop": mean_drop,
            "max_performance_drop": max_drop,
        }

    approx_kl = _unique_update_values(training_log, "approx_kl")
    clip_fraction = _unique_update_values(training_log, "clip_fraction")
    actor_lr = _unique_update_values(training_log, "actor_lr")

    if approx_kl.size > 0:
        metrics["kl_stability"] = {
            "mean_kl": float(np.mean(approx_kl)),
            "kl_std": float(np.std(approx_kl)),
            "mean_abs_kl_error": float(np.mean(np.abs(approx_kl - target_kl))),
            "kl_overshoot_rate": float(np.mean(approx_kl > 2.0 * target_kl)),
            "clip_fraction_mean": float(np.mean(clip_fraction)) if clip_fraction.size > 0 else None,
            "clip_fraction_std": float(np.std(clip_fraction)) if clip_fraction.size > 0 else None,
            "actor_lr_mean": float(np.mean(actor_lr)) if actor_lr.size > 0 else None,
            "actor_lr_min": float(np.min(actor_lr)) if actor_lr.size > 0 else None,
            "actor_lr_max": float(np.max(actor_lr)) if actor_lr.size > 0 else None,
            "num_ppo_updates_measured": int(approx_kl.size),
        }

    return metrics


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _append_episode_diagnostics(training_log: Dict, info: Dict) -> None:
    training_log["avg_distance_to_fire"].append(float(info.get("avg_distance_to_fire", 0.0)))
    training_log["first_heat_step"].append(int(info.get("first_heat_step", -1)))
    training_log["first_boundary_step"].append(int(info.get("first_boundary_step", -1)))
    training_log["spawn_modes"].append(list(info.get("spawn_modes", [])))
    training_log["reward_breakdown"].append(info.get("reward_breakdown") or {})


class ActorNetwork(nn.Module):
    def __init__(self, local_obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(local_obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(local_obs)))

        identity = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + identity

        identity = x
        x = F.relu(self.ln3(self.fc3(x)))
        x = x + identity

        identity = x
        x = F.relu(self.ln4(self.fc4(x)))
        x = x + identity

        x = F.relu(self.fc5(x))
        return self.action_head(x)

    def get_action_probs(self, local_obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(local_obs))


class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 160)
        self.ln4 = nn.LayerNorm(160)
        self.value_head = nn.Linear(160, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(global_state)))

        identity = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + identity

        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        return self.value_head(x)


class ReplayBuffer:
    def __init__(self):
        self.local_obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store(self, local_obs, global_state, actions, log_probs, rewards, done):
        self.local_obs.append(local_obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)

    def clear(self):
        self.local_obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def get(self):
        return self.local_obs, self.global_states, self.actions, self.log_probs, self.rewards, self.dones

    def __len__(self):
        return len(self.rewards)


class CurriculumManager:
    PERCENTILE_LADDER = [None, 1.0, 2.5, 5.0]
    STAGE3_TARGET_MIN_EPS = [100, 200, 350, 0]
    STAGE3_NEAR_SPAWN_INIT = 0.25
    # 方案 C: 能力绑定阶梯式退火，替代旧的线性退火
    STAGE3_NEAR_LADDER = [0.25, 0.15, 0.05, 0.0]
    STAGE3_NEAR_MIN_EPS = [80, 100, 120]  # 每级最少回合数
    # 每级退火的能力门槛: (success_rate, max_zero_timeout_rate, min_coverage)
    STAGE3_NEAR_GATES = [
        (0.30, 0.20, 0.25),   # 0.25 -> 0.15
        (0.40, 0.15, 0.35),   # 0.15 -> 0.05
        (0.50, 0.10, 0.45),   # 0.05 -> 0.00
    ]
    TERMINAL_FOCUS_EPISODES = 300  # 最后 300 回合强制评估条件

    def __init__(self, final_init_area_percent: float = 5.0, stage3_final_target: float = 0.60):
        self.current_stage = 1
        self.stage_episodes = {1: 0, 2: 0, 3: 0}
        self.stage_success_rates = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self.stage_coverages = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self.stage_zero_timeout_rates = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self.stage_thresholds = {1: 0.40, 2: 0.42}
        self.stage_zero_timeout_thresholds = {1: 0.50, 2: 0.35, 3: 0.20}
        self.stage_min_episodes = {1: 150, 2: 300}
        self.stage_max_episodes = {1: 220, 2: 450}
        self.force_advance_min_coverage = {1: 0.03, 2: 0.15}
        self._pct_idx = 0
        self._pct_eps = 0
        self._pct_min_eps = 80
        self._pct_advance_threshold = 0.35
        self._s3_target_idx = 0
        self._s3_target_eps = 0
        self._s3_near_idx = 0
        self._near_eps = 0
        self.PERCENTILE_LADDER = [None, 1.0, 2.5, float(final_init_area_percent)]
        self.STAGE3_TARGET_LADDER = [0.20, 0.35, 0.50, float(stage3_final_target)]
        self._terminal_focus_active = False

    @property
    def current_init_percentile(self):
        return self.PERCENTILE_LADDER[self._pct_idx]

    @property
    def current_stage3_target(self) -> float:
        return self.STAGE3_TARGET_LADDER[self._s3_target_idx]

    @property
    def stage3_near_prob(self) -> float:
        if self.current_stage != 3:
            return self.STAGE3_NEAR_SPAWN_INIT
        return self.STAGE3_NEAR_LADDER[self._s3_near_idx]

    def update(self, success: bool, coverage: float, zero_coverage_timeout: bool = False) -> int:
        stage = self.current_stage
        self.stage_episodes[stage] += 1
        self.stage_success_rates[stage].append(1.0 if success else 0.0)
        self.stage_coverages[stage].append(float(coverage))
        self.stage_zero_timeout_rates[stage].append(1.0 if zero_coverage_timeout else 0.0)

        if stage == 1:
            self._pct_eps += 1
            self._try_advance_percentile()
        if stage == 3:
            self._s3_target_eps += 1
            self._near_eps += 1
            self._try_advance_stage3_target()
            self._try_advance_near_prob()
            return self.current_stage

        success_rate = float(np.mean(self.stage_success_rates[stage])) if self.stage_success_rates[stage] else 0.0
        avg_coverage = float(np.mean(self.stage_coverages[stage])) if self.stage_coverages[stage] else 0.0
        zero_timeout_rate = (
            float(np.mean(self.stage_zero_timeout_rates[stage]))
            if self.stage_zero_timeout_rates[stage]
            else 0.0
        )
        coverage_ready = avg_coverage >= self.force_advance_min_coverage[stage]
        zero_timeout_ready = zero_timeout_rate <= self.stage_zero_timeout_thresholds[stage]
        capability_ready = (
            self.stage_episodes[stage] >= self.stage_min_episodes[stage]
            and success_rate >= self.stage_thresholds[stage]
            and coverage_ready
            and zero_timeout_ready
        )
        force_advance = (
            self.stage_episodes[stage] >= self.stage_max_episodes[stage]
            and stage == 1
            and coverage_ready
            and zero_timeout_ready
        )

        if capability_ready or force_advance:
            old_stage = self.current_stage
            self.current_stage += 1
            print(
                f"\n课程阶段 {old_stage} -> {self.current_stage} | "
                f"本阶段回合={self.stage_episodes[old_stage]} | "
                f"成功率={success_rate * 100:.1f}% | 覆盖率={avg_coverage * 100:.1f}% | "
                f"零覆盖超时={zero_timeout_rate * 100:.1f}%"
            )

        return self.current_stage

    def _try_advance_percentile(self):
        if self._pct_idx >= len(self.PERCENTILE_LADDER) - 1:
            return
        success_rate = float(np.mean(self.stage_success_rates[1])) if self.stage_success_rates[1] else 0.0
        if self._pct_eps >= self._pct_min_eps and success_rate >= self._pct_advance_threshold:
            self._pct_idx += 1
            self._pct_eps = 0
            print(
                f"\n  [area curriculum] init_area_percent -> "
                f"{self.current_init_percentile} | stage1_success={success_rate * 100:.1f}%"
            )

    def _try_advance_stage3_target(self):
        if self._s3_target_idx >= len(self.STAGE3_TARGET_LADDER) - 1:
            return
        min_eps = self.STAGE3_TARGET_MIN_EPS[self._s3_target_idx]
        avg_coverage = float(np.mean(self.stage_coverages[3])) if self.stage_coverages[3] else 0.0
        success_rate = float(np.mean(self.stage_success_rates[3])) if self.stage_success_rates[3] else 0.0
        zero_timeout_rate = (
            float(np.mean(self.stage_zero_timeout_rates[3]))
            if self.stage_zero_timeout_rates[3]
            else 0.0
        )
        current_target = self.STAGE3_TARGET_LADDER[self._s3_target_idx]
        # 方案 C: 更严格的能力门槛
        if (
            self._s3_target_eps >= min_eps
            and avg_coverage >= current_target * 0.85
            and success_rate >= 0.50
            and zero_timeout_rate <= 0.15
        ):
            self._s3_target_idx += 1
            self._s3_target_eps = 0
            print(
                f"\n  [stage3 curriculum] target {current_target:.0%} -> "
                f"{self.current_stage3_target:.0%} | avg_coverage={avg_coverage * 100:.1f}% | "
                f"success={success_rate * 100:.1f}% | zero_timeout={zero_timeout_rate * 100:.1f}%"
            )

    def _try_advance_near_prob(self):
        """方案 C: 能力绑定阶梯式 near_prob 退火，且不超过 target 进度。"""
        if self._s3_near_idx >= len(self.STAGE3_NEAR_LADDER) - 1:
            return
        # 关键约束: near_prob 退火不超前于 target 推进
        if self._s3_near_idx >= self._s3_target_idx:
            return
        min_eps = self.STAGE3_NEAR_MIN_EPS[self._s3_near_idx]
        if self._near_eps < min_eps:
            return
        avg_coverage = float(np.mean(self.stage_coverages[3])) if self.stage_coverages[3] else 0.0
        success_rate = float(np.mean(self.stage_success_rates[3])) if self.stage_success_rates[3] else 0.0
        zero_timeout_rate = (
            float(np.mean(self.stage_zero_timeout_rates[3]))
            if self.stage_zero_timeout_rates[3]
            else 0.0
        )
        req_sr, req_zt, req_cov = self.STAGE3_NEAR_GATES[self._s3_near_idx]
        if success_rate >= req_sr and zero_timeout_rate <= req_zt and avg_coverage >= req_cov:
            old_prob = self.STAGE3_NEAR_LADDER[self._s3_near_idx]
            self._s3_near_idx += 1
            self._near_eps = 0
            new_prob = self.STAGE3_NEAR_LADDER[self._s3_near_idx]
            print(
                f"\n  [near curriculum] near_prob {old_prob:.2f} -> {new_prob:.2f} | "
                f"success={success_rate * 100:.1f}% | zero_timeout={zero_timeout_rate * 100:.1f}% | "
                f"coverage={avg_coverage * 100:.1f}%"
            )

    def get_stage_info(self) -> Dict:
        stage = self.current_stage
        success_rate = float(np.mean(self.stage_success_rates[stage])) if self.stage_success_rates[stage] else 0.0
        return {
            "stage": stage,
            "episodes": self.stage_episodes[stage],
            "success_rate": success_rate,
            "total_episodes": sum(self.stage_episodes.values()),
            "init_area_percent": self.current_init_percentile,
            "stage3_target": self.current_stage3_target if stage == 3 else None,
            "stage3_near_prob": self.stage3_near_prob if stage == 3 else None,
        }

    def activate_terminal_focus(self):
        """强制切换到评估条件: target=最终值, near_prob=0.0"""
        self._s3_target_idx = len(self.STAGE3_TARGET_LADDER) - 1
        self._s3_near_idx = len(self.STAGE3_NEAR_LADDER) - 1


class CTDE_PPO_Agent:
    def __init__(
        self,
        local_obs_dim: int,
        global_state_dim: int,
        action_dim: int,
        num_agents: int,
        actor_lr: float = 2e-4,
        critic_lr: float = 5e-4,
        lr_adapt_mode: str = "fixed",
        target_kl: float = 0.010,
        actor_lr_min: float = 2e-5,
        actor_lr_max: float = 4e-4,
        kl_ema_beta: float = 0.9,
        kl_lr_alpha: float = 0.1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 4096,
        device: str = "auto",
    ):
        self.num_agents = int(num_agents)
        self.lr_adapt_mode = str(lr_adapt_mode).lower()
        if self.lr_adapt_mode not in {"fixed", "kl"}:
            raise ValueError("lr_adapt_mode must be 'fixed' or 'kl'")
        self.target_kl = max(1e-8, float(target_kl))
        self.actor_lr_min = max(1e-12, float(actor_lr_min))
        self.actor_lr_max = max(self.actor_lr_min, float(actor_lr_max))
        self.kl_ema_beta = float(np.clip(kl_ema_beta, 0.0, 0.999))
        self.kl_lr_alpha = max(0.0, float(kl_lr_alpha))
        self.kl_ema = None
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.mini_batch_size = max(512, self.batch_size // 8)
        self.min_update_batch_size = max(512, self.batch_size // 4)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = ActorNetwork(local_obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(global_state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.buffer = ReplayBuffer()
        self.training_step = 0

        print(
            f"CTDE-PPO 基线已初始化 | 设备={self.device} | "
            f"本地观测={local_obs_dim} | 全局状态={global_state_dim} | "
            f"lr_adapt_mode={self.lr_adapt_mode}"
        )

    def _set_actor_lr(self, lr: float):
        lr = float(np.clip(lr, self.actor_lr_min, self.actor_lr_max))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr

    def _update_kl_ema(self, mean_kl: float) -> float:
        mean_kl = float(mean_kl)
        if self.kl_ema is None:
            self.kl_ema = mean_kl
        else:
            self.kl_ema = self.kl_ema_beta * self.kl_ema + (1.0 - self.kl_ema_beta) * mean_kl
        return float(self.kl_ema)

    def _adapt_actor_lr_by_kl(self, mean_kl: float) -> str:
        kl_ema = self._update_kl_ema(mean_kl)
        current_lr = float(self.actor_optimizer.param_groups[0]["lr"])
        lr_factor = float(np.exp(-self.kl_lr_alpha * (kl_ema / self.target_kl - 1.0)))
        new_lr = float(np.clip(current_lr * lr_factor, self.actor_lr_min, self.actor_lr_max))
        self._set_actor_lr(new_lr)

        if np.isclose(new_lr, current_lr):
            return "keep"
        if new_lr < current_lr:
            return "down"
        return "up"

    def select_actions(self, local_obs: List[np.ndarray]) -> Tuple[List[int], List[float]]:
        local_obs_tensor = torch.FloatTensor(np.array(local_obs)).to(self.device)
        with torch.no_grad():
            action_probs = self.actor.get_action_probs(local_obs_tensor)
            actions = action_probs.sample()
            log_probs = action_probs.log_prob(actions)
        return actions.cpu().numpy().tolist(), log_probs.cpu().numpy().tolist()

    def select_actions_deterministic(self, local_obs: List[np.ndarray]) -> List[int]:
        local_obs_tensor = torch.FloatTensor(np.array(local_obs)).to(self.device)
        with torch.no_grad():
            logits = self.actor(local_obs_tensor)
            actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy().tolist()

    def store_transition(self, local_obs, global_state, actions, log_probs, rewards, done):
        self.buffer.store(local_obs, global_state, actions, log_probs, rewards, done)

    def compute_gae(self, rewards_list: List[List[float]], dones: List[bool], global_states: np.ndarray):
        global_states_tensor = torch.FloatTensor(global_states).to(self.device)
        with torch.no_grad():
            values = self.critic(global_states_tensor).squeeze(-1)

        team_rewards = [float(np.mean(rewards)) for rewards in rewards_list]
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(team_rewards))):
            if t == len(team_rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = team_rewards[t] + self.gamma * next_value * (1 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[t])) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def update(self, force: bool = False) -> Dict[str, float]:
        required_batch = self.min_update_batch_size if force else self.batch_size
        buffer_size = len(self.buffer)
        if buffer_size < required_batch:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        local_obs_list, global_states, actions_list, old_log_probs_list, rewards_list, dones = self.buffer.get()
        global_states_np = np.array(global_states)
        advantages, returns = self.compute_gae(rewards_list, dones, global_states_np)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        global_states_tensor = torch.FloatTensor(global_states_np).to(self.device)
        local_obs_tensor = torch.FloatTensor(np.array(local_obs_list)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions_list)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs_list)).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        update_steps = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(buffer_size, device=self.device)
            for start in range(0, buffer_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_global_states = global_states_tensor[mb_indices]
                mb_returns = returns[mb_indices]
                values_pred = self.critic(mb_global_states).squeeze(-1)
                critic_loss = self.value_coef * F.mse_loss(values_pred, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                mb_advantages = advantages[mb_indices]
                mb_local_obs = local_obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]

                obs_dim = mb_local_obs.shape[-1]
                flat_obs = mb_local_obs.view(-1, obs_dim)
                flat_actions = mb_actions.view(-1)
                flat_old_log_probs = mb_old_log_probs.view(-1)
                flat_advantages = mb_advantages.unsqueeze(1).expand(-1, self.num_agents).reshape(-1)

                action_probs = self.actor.get_action_probs(flat_obs)
                new_log_probs = action_probs.log_prob(flat_actions)
                entropy = action_probs.entropy().mean()

                log_ratio = new_log_probs - flat_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * flat_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * flat_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                total_approx_kl += float(approx_kl.item())
                total_clip_fraction += float(clip_fraction.item())
                update_steps += 1

        self.buffer.clear()
        self.training_step += 1

        denom = max(update_steps, 1)
        mean_kl = total_approx_kl / denom
        if self.lr_adapt_mode == "kl":
            kl_lr_action = self._adapt_actor_lr_by_kl(mean_kl)
        else:
            self._update_kl_ema(mean_kl)
            kl_lr_action = "fixed"

        return {
            "actor_loss": total_actor_loss / denom,
            "critic_loss": total_critic_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": mean_kl,
            "kl_ema": self.kl_ema if self.kl_ema is not None else mean_kl,
            "kl_lr_action": kl_lr_action,
            "clip_fraction": total_clip_fraction / denom,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "entropy_coef": self.entropy_coef,
        }

    def save(self, path: str):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "kl_ema": self.kl_ema,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.training_step = int(checkpoint["training_step"])
        self.kl_ema = checkpoint.get("kl_ema")


def _make_output_dir(config: Dict) -> str:
    if config.get("output_dir"):
        output_dir = os.path.abspath(config["output_dir"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = config.get("output_root_dir", "./outputs")
        if not os.path.isabs(output_root):
            output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_root)
        output_dir = os.path.join(os.path.abspath(output_root), timestamp, RESULTS_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_source_snapshot(output_dir: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.abspath(output_dir))
    source_dir = os.path.join(package_dir, SOURCE_DIR_NAME)
    os.makedirs(source_dir, exist_ok=True)

    for filename in [
        os.path.basename(__file__),
        "rl_environment_baseline.py",
        "\u4fe1\u606f\u8f6c\u6362.py",
    ]:
        src = os.path.join(script_dir, filename)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"source file not found: {src}")
        shutil.copy2(src, os.path.join(source_dir, filename))

    return source_dir


def _run_figure_scripts(
    result_dir: str,
    config: Dict,
    include_generalization: bool = True,
    out_root: str = None,
    seed_filter: int = None,
    aggregate_seeds: bool = False,
) -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "outputs")
    figure_paths = {}
    if out_root is None:
        out_root = os.path.join(os.path.abspath(result_dir), "figures")
    run_filter = f"seed{int(seed_filter)}" if seed_filter is not None else None

    training_script = os.path.join(outputs_dir, "make_training_figures.py")
    if os.path.isfile(training_script):
        cmd = [
            sys.executable,
            training_script,
            "--results-dir",
            os.path.abspath(result_dir),
            "--window",
            str(config["figure_window"]),
            "--dpi",
            str(config["figure_dpi"]),
            "--max-steps",
            str(config["max_steps"]),
            "--out-dir",
            os.path.join(out_root, "training_figures"),
        ]
        if run_filter is not None:
            cmd.extend(["--run-filter", run_filter])
        if aggregate_seeds:
            cmd.append("--aggregate-seeds")
        print("\n正在生成训练图表...")
        subprocess.run(cmd, check=True)
        figure_paths["training_figures"] = os.path.join(out_root, "training_figures")
    else:
        print(f"未找到训练图表脚本: {training_script}")

    if include_generalization:
        generalization_script = os.path.join(outputs_dir, "make_generalization_figures.py")
        if os.path.isfile(generalization_script):
            cmd = [
                sys.executable,
                generalization_script,
                "--results-dir",
                os.path.abspath(result_dir),
                "--window",
                "10",
                "--dpi",
                str(config["figure_dpi"]),
                "--max-steps",
                str(config["max_steps"]),
                "--out-dir",
                os.path.join(out_root, "generalization_figures"),
            ]
            if run_filter is not None:
                cmd.extend(["--run-filter", run_filter])
            if aggregate_seeds:
                cmd.append("--aggregate-seeds")
            print("\n正在生成泛化评估图表...")
            subprocess.run(cmd, check=True)
            figure_paths["generalization_figures"] = os.path.join(out_root, "generalization_figures")
        else:
            print(f"未找到泛化评估图表脚本: {generalization_script}")

    return figure_paths


def _resolve_dataset_scene_keys(config: Dict) -> DatasetIndex:
    dataset_index = DatasetIndex(config["data_dir"])
    if config.get("train_scene_keys") is None:
        config["train_scene_keys"] = dataset_index.scene_keys(config["train_split"])
    if config.get("eval_scene_keys") is None:
        config["eval_scene_keys"] = dataset_index.scene_keys(config["eval_split"])
    return dataset_index


def _build_experiment_metadata(
    config: Dict,
    dataset_index: DatasetIndex,
    env_info: Dict = None,
) -> Dict:
    env_info = {} if env_info is None else dict(env_info)
    split_counts = {
        split: len(dataset_index.splits.get(split, []))
        for split in ["train", "validation", "generalization", "stress"]
    }
    return {
        "dataset_index_version": dataset_index.index.get("version"),
        "scene_split_counts": split_counts,
        "observation_profile": config["observation_profile"],
        "reward_profile": config["reward_profile"],
        "observation_profile_dims": copy.deepcopy(config["observation_profile_dims"]),
        "norm_params_source": config["norm_params_source"],
        "use_scene_uav_params": bool(config["use_metadata_uav_params"]),
        "use_metadata_uav_params": bool(config["use_metadata_uav_params"]),
        "vision_radius": int(env_info.get("vision_radius", config["vision_radius"])),
        "configured_vision_radius": int(config["vision_radius"]),
        "sensor_radius_cells": env_info.get("sensor_radius_cells"),
        "max_steps": int(env_info.get("max_steps", config["max_steps"])),
        "configured_max_steps": int(config["max_steps"]),
        "scene_key": env_info.get("scene_key"),
    }


def make_eval_config(
    base_config: Dict,
    split: str,
    episodes_per_scene: int,
    stages: List[int],
) -> Dict:
    eval_config = copy.deepcopy(base_config)
    eval_config["eval_split"] = str(split).lower()
    eval_config["eval_scene_keys"] = None
    eval_config["eval_episodes_per_scene"] = int(episodes_per_scene)
    eval_config["eval_stages"] = [int(stage) for stage in stages]
    return eval_config


def evaluate_preserving_rng(agent: "CTDE_PPO_Agent", eval_config: Dict) -> Dict:
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return evaluate(agent, eval_config)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _stage_summary(results: Dict, stage: int) -> Dict:
    summary = dict(results.get(int(stage), results.get(str(stage), {})))
    summary.pop("records", None)
    return summary


def _after_train_eval_checkpoints(config: Dict, best_model_paths: Dict) -> List[Tuple[str, str]]:
    best_val_path = best_model_paths.get("best_val")
    if (
        config["evaluate_best_val_after_train"]
        and best_val_path
        and os.path.exists(best_val_path)
    ):
        return [("best_val", best_val_path)]
    return []


def _eval_summary_stage(eval_summary: Dict, split: str, stage_key: str) -> Dict:
    best_val_stage = (
        eval_summary.get("best_val", {})
        .get("splits", {})
        .get(split, {})
        .get("stages", {})
        .get(str(stage_key), {})
    )
    if best_val_stage:
        return best_val_stage
    return (
        eval_summary.get("final", {})
        .get(split, {})
        .get("stages", {})
        .get(str(stage_key), {})
    )


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=False)


def _thermal_health_failures(records: List[Dict]) -> List[str]:
    failures = []
    for record in records:
        split = str(record.get("split", "?"))
        scene_key = str(record.get("scene_key", "?"))
        for metric, limit in THERMAL_HEALTH_LIMITS.items():
            value = float(record.get(metric, 0.0))
            if value > limit:
                failures.append(
                    f"{split}/{scene_key} {metric}={value:.3f} > {limit:.3f}"
                )
    return failures


def _assert_thermal_health_ok(records: List[Dict]) -> None:
    failures = _thermal_health_failures(records)
    if failures:
        shown = "; ".join(failures[:8])
        remaining = len(failures) - min(len(failures), 8)
        suffix = f"; ... {remaining} more" if remaining > 0 else ""
        raise RuntimeError(
            "Thermal health check failed before training: " + shown + suffix
        )


def _collect_thermal_health(
    data_dir: str,
    dataset_index: DatasetIndex,
    splits: List[str],
    init_percentile,
    init_area_percent,
    scene_keys_by_split: Dict[str, List[str]] = None,
) -> List[Dict]:
    scene_manager = SceneManager(data_dir)
    scene_keys_by_split = scene_keys_by_split or {}
    records = []
    for split in splits:
        split = dataset_index.normalize_mode(split)
        scene_keys = scene_keys_by_split.get(split) or dataset_index.scene_keys(split)
        for scene_key in scene_keys:
            scene = scene_manager.get_specific_scene(scene_key)
            scene.initialize_training_boundary(
                init_percentile=init_percentile,
                init_area_percent=init_area_percent,
            )
            scene._compute_thermal_field()
            record = scene.diagnose_thermal_health()
            record["split"] = split
            record["scene_key"] = scene_key
            records.append(record)
    return records


def train(config: Dict = None):
    config = normalize_training_config(config)

    output_dir = _make_output_dir(config)
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    console_log_path = os.path.join(output_dir, CONSOLE_LOG_NAME)
    setup_console_tee(console_log_path, mode="w")

    set_seed(config["seed"])

    dataset_index = _resolve_dataset_scene_keys(config)
    preflight_counts = validate_scene_boundaries(
        config["data_dir"],
        splits=["train", "validation", "generalization", "stress"],
        init_percentile=config["init_percentile"],
        init_area_percent=config["init_area_percent"],
        verbose=True,
    )
    thermal_health_records = _collect_thermal_health(
        config["data_dir"],
        dataset_index,
        splits=["train", "validation", "generalization", "stress"],
        init_percentile=config["init_percentile"],
        init_area_percent=config["init_area_percent"],
        scene_keys_by_split={config["train_split"]: config["train_scene_keys"]},
    )
    preflight_payload = _build_experiment_metadata(config, dataset_index)
    preflight_payload["boundary_validation"] = preflight_counts
    preflight_payload["thermal_health"] = {
        "limits": dict(THERMAL_HEALTH_LIMITS),
        "records": thermal_health_records,
        "failures": _thermal_health_failures(thermal_health_records),
    }
    _write_json(os.path.join(log_dir, "dataset_preflight.json"), preflight_payload)
    _assert_thermal_health_ok(thermal_health_records)

    print("\n" + "=" * 70)
    print("CTDE-PPO 基线训练开始")
    print("=" * 70)
    print(f"输出目录={output_dir}")
    print(f"随机种子={config['seed']}")
    print(
        f"学习率策略={config['lr_adapt_mode']} | "
        f"target_kl={config['target_kl']:.4f} | actor_lr={config['actor_lr']:.2e} | "
        f"kl_lr_alpha={config['kl_lr_alpha']:.3f}"
    )
    print(
        f"observation_profile={config['observation_profile']} | "
        f"reward_profile={config['reward_profile']} | "
        f"norm_params_source={config['norm_params_source']}"
    )
    print(f"初始位置百分位={config['init_percentile']}")

    curriculum = CurriculumManager(
        final_init_area_percent=config["init_area_percent"],
        stage3_final_target=config["stage3_success_target"],
    )
    env = FireSearchBaselineEnvironment(
        data_dir=config["data_dir"],
        num_drones=config["num_drones"],
        vision_radius=config["vision_radius"],
        max_steps=config["max_steps"],
        use_metadata_uav_params=config["use_metadata_uav_params"],
        observation_profile=config["observation_profile"],
        reward_profile=config["reward_profile"],
        curriculum_stage=1,
        mode=config["train_split"],
        scene_keys=config["train_scene_keys"],
        init_percentile=config["init_percentile"],
        init_area_percent=curriculum.current_init_percentile,
        stage2_target=config["stage2_success_target"],
        stage3_target=curriculum.current_stage3_target,
        stage3_near_prob=curriculum.stage3_near_prob,
    )
    experiment_metadata = _build_experiment_metadata(
        config, dataset_index, env.get_env_info()
    )
    config["experiment_metadata"] = experiment_metadata
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config), f, indent=2, ensure_ascii=False)

    agent = CTDE_PPO_Agent(
        local_obs_dim=env.local_obs_dim,
        global_state_dim=env.global_state_dim,
        action_dim=env.num_actions,
        num_agents=env.num_drones,
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        lr_adapt_mode=config["lr_adapt_mode"],
        target_kl=config["target_kl"],
        actor_lr_min=config["actor_lr_min"],
        actor_lr_max=config["actor_lr_max"],
        kl_ema_beta=config["kl_ema_beta"],
        kl_lr_alpha=config["kl_lr_alpha"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_coef"],
        value_coef=config["value_coef"],
        max_grad_norm=config["max_grad_norm"],
        ppo_epochs=config["ppo_epochs"],
        batch_size=config["batch_size"],
    )

    rolling_rewards = deque(maxlen=100)
    rolling_lengths = deque(maxlen=100)
    rolling_coverages = deque(maxlen=100)
    rolling_success = deque(maxlen=100)
    rolling_task_scores = deque(maxlen=100)
    rolling_timeouts = deque(maxlen=100)
    rolling_zero_timeouts = deque(maxlen=100)

    training_log = {
        "experiment": experiment_metadata,
        "dataset_index_version": experiment_metadata["dataset_index_version"],
        "scene_split_counts": experiment_metadata["scene_split_counts"],
        "observation_profile": config["observation_profile"],
        "reward_profile": config["reward_profile"],
        "norm_params_source": config["norm_params_source"],
        "use_scene_uav_params": bool(config["use_metadata_uav_params"]),
        "episodes": [],
        "rewards": [],
        "task_scores": [],
        "lengths": [],
        "coverages": [],
        "success": [],
        "done_reasons": [],
        "timeout": [],
        "zero_coverage_timeout": [],
        "avg_distance_to_fire": [],
        "first_heat_step": [],
        "first_boundary_step": [],
        "spawn_modes": [],
        "reward_breakdown": [],
        "stage": [],
        "scene_ids": [],
        "scene_keys": [],
        "vision_radius": [],
        "sensor_radius_cells": [],
        "max_steps": [],
        "total_steps": [],
        "ppo_updates": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "kl_ema": [],
        "kl_lr_action": [],
        "clip_fraction": [],
        "actor_lr": [],
        "critic_lr": [],
        "init_area_percent": [],
        "stage3_target": [],
        "stage3_near_prob": [],
        "terminal_focus": [],
    }

    validation_log = {
        "episodes": [],
        "stage": [],
        "train_task_score": [],
        "val_mean_task_score": [],
        "val_mean_coverage": [],
        "val_success_rate": [],
        "val_mean_length": [],
        "val_timeout_rate": [],
        "val_zero_coverage_timeout_rate": [],
        "generalization_gap": [],
        "is_best_val": [],
    }

    start_time = time.time()
    total_steps = 0
    best_task_score = -float("inf")
    best_val_model_score = -float("inf")
    best_model_paths = {"best": None, "best_train": None, "best_val": None, "final": None}
    update_info = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "kl_ema": 0.0,
        "kl_lr_action": "fixed" if config["lr_adapt_mode"] == "fixed" else "keep",
        "clip_fraction": 0.0,
        "actor_lr": config["actor_lr"],
        "critic_lr": config["critic_lr"],
    }

    for episode in range(1, config["total_episodes"] + 1):
        # 终末专注: 最后 N 回合强制评估条件
        remaining = config["total_episodes"] - episode
        if (
            remaining <= CurriculumManager.TERMINAL_FOCUS_EPISODES
            and curriculum.current_stage == 3
            and not curriculum._terminal_focus_active
        ):
            curriculum.activate_terminal_focus()
            curriculum._terminal_focus_active = True
            # 立即同步 env 参数，避免首回合使用旧值
            env.stage_targets[3] = curriculum.current_stage3_target
            env.stage3_near_prob = curriculum.stage3_near_prob
            print(
                f"\n  [terminal focus] 剩余{remaining}回合, "
                f"强制 target={curriculum.current_stage3_target:.2f}, "
                f"near_prob={curriculum.stage3_near_prob:.2f}"
            )
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            local_obs = obs["local_obs"]
            global_state = obs["global_state"]
            actions, log_probs = agent.select_actions(local_obs)
            next_obs, rewards, done, info = env.step(actions)
            agent.store_transition(local_obs, global_state, actions, log_probs, rewards, done)

            episode_reward += float(sum(rewards))
            episode_length += 1
            total_steps += 1
            obs = next_obs

        if len(agent.buffer) >= config["batch_size"]:
            update_info = agent.update()

        success = info["done_reason"] == "mission_complete"
        timeout = info["done_reason"] == "max_steps_reached"
        zero_timeout = bool(info.get("zero_coverage_timeout", False))
        task_score = _task_score(info["boundary_coverage"], success, episode_length, config["max_steps"])

        rolling_rewards.append(episode_reward)
        rolling_lengths.append(episode_length)
        rolling_coverages.append(info["boundary_coverage"])
        rolling_success.append(1.0 if success else 0.0)
        rolling_task_scores.append(task_score)
        rolling_timeouts.append(1.0 if timeout else 0.0)
        rolling_zero_timeouts.append(1.0 if zero_timeout else 0.0)

        training_log["episodes"].append(episode)
        training_log["rewards"].append(episode_reward)
        training_log["task_scores"].append(task_score)
        training_log["lengths"].append(episode_length)
        training_log["coverages"].append(info["boundary_coverage"])
        training_log["success"].append(1 if success else 0)
        training_log["done_reasons"].append(info.get("done_reason", "other"))
        training_log["timeout"].append(1 if timeout else 0)
        training_log["zero_coverage_timeout"].append(1 if zero_timeout else 0)
        _append_episode_diagnostics(training_log, info)
        training_log["stage"].append(curriculum.current_stage)
        training_log["scene_ids"].append(info["scene_id"])
        training_log["scene_keys"].append(info.get("scene_key", str(info["scene_id"])))
        training_log["vision_radius"].append(int(info.get("vision_radius", config["vision_radius"])))
        training_log["sensor_radius_cells"].append(info.get("sensor_radius_cells"))
        training_log["max_steps"].append(int(info.get("max_steps", config["max_steps"])))
        training_log["total_steps"].append(total_steps)
        training_log["ppo_updates"].append(agent.training_step)
        training_log["actor_loss"].append(update_info.get("actor_loss", 0.0))
        training_log["critic_loss"].append(update_info.get("critic_loss", 0.0))
        training_log["entropy"].append(update_info.get("entropy", 0.0))
        training_log["approx_kl"].append(update_info.get("approx_kl", 0.0))
        training_log["kl_ema"].append(update_info.get("kl_ema", update_info.get("approx_kl", 0.0)))
        training_log["kl_lr_action"].append(update_info.get("kl_lr_action", "fixed"))
        training_log["clip_fraction"].append(update_info.get("clip_fraction", 0.0))
        training_log["actor_lr"].append(update_info.get("actor_lr", config["actor_lr"]))
        training_log["critic_lr"].append(update_info.get("critic_lr", config["critic_lr"]))
        training_log["init_area_percent"].append(env.init_area_percent)
        training_log["stage3_target"].append(env.stage_targets[3])
        training_log["stage3_near_prob"].append(env.stage3_near_prob)
        training_log["terminal_focus"].append(
            1 if curriculum._terminal_focus_active else 0
        )

        old_stage = curriculum.current_stage
        new_stage = curriculum.update(
            success,
            info["boundary_coverage"],
            zero_coverage_timeout=zero_timeout,
        )
        next_area_percent = curriculum.current_init_percentile
        next_stage3_target = curriculum.current_stage3_target
        next_stage3_near_prob = curriculum.stage3_near_prob
        difficulty_changed = (
            next_area_percent != env.init_area_percent
            or next_stage3_target != env.stage_targets[3]
            or next_stage3_near_prob != env.stage3_near_prob
        )
        if new_stage != env.curriculum_stage or difficulty_changed:
            if len(agent.buffer) >= agent.min_update_batch_size:
                pending_update_info = agent.update(force=True)
                if pending_update_info.get("actor_loss", 0.0) != 0.0:
                    update_info = pending_update_info
            else:
                agent.buffer.clear()
        if next_area_percent != env.init_area_percent:
            env.init_area_percent = next_area_percent
            env.init_percentile = next_area_percent
            print(f"  [curriculum] env.init_area_percent -> {next_area_percent}")
        if next_stage3_target != env.stage_targets[3]:
            env.stage_targets[3] = next_stage3_target
            print(f"  [curriculum] env.stage3_target -> {next_stage3_target:.2f}")
        if next_stage3_near_prob != env.stage3_near_prob:
            env.stage3_near_prob = next_stage3_near_prob
            print(f"  [curriculum] env.stage3_near_prob -> {next_stage3_near_prob:.3f}")
        if new_stage != env.curriculum_stage:
            env.set_curriculum_stage(new_stage)
            print(f"切换到阶段 {new_stage} 前已处理阶段 {old_stage} 的缓存数据")

        if episode % config["log_interval"] == 0:
            stage_info = curriculum.get_stage_info()
            print(
                f"回合 {episode:4d} | 场景={info.get('scene_key', info['scene_id'])} | 阶段={stage_info['stage']} | "
                f"奖励={np.mean(rolling_rewards):7.1f} | 步数={np.mean(rolling_lengths):5.1f} | "
                f"覆盖率={np.mean(rolling_coverages) * 100:5.1f}% | "
                f"成功率={np.mean(rolling_success) * 100:4.0f}% | "
                f"任务得分={np.mean(rolling_task_scores) * 100:5.1f}% | "
                f"超时率={np.mean(rolling_timeouts) * 100:4.0f}% | "
                f"零覆盖超时={np.mean(rolling_zero_timeouts) * 100:4.0f}% | "
                f"KL={update_info.get('approx_kl', 0.0):.4f} | "
                f"KLema={update_info.get('kl_ema', 0.0):.4f} | "
                f"clip={update_info.get('clip_fraction', 0.0):.3f} | "
                f"lr={update_info.get('actor_lr', config['actor_lr']):.2e} | "
                f"lr_action={update_info.get('kl_lr_action', 'fixed')} | "
                f"状态={info['done_reason']}"
            )

        if episode % config["validation_interval"] == 0:
            val_config = make_eval_config(
                config,
                config["validation_split"],
                config["validation_episodes_per_scene"],
                [curriculum.current_stage],
            )
            print("\n" + "=" * 70)
            print(
                f"验证评估 | 回合={episode} | "
                f"阶段={curriculum.current_stage} | "
                f"每场景回合={config['validation_episodes_per_scene']}"
            )
            print("=" * 70)
            validation_results = evaluate_preserving_rng(agent, val_config)
            val_summary = validation_results[int(curriculum.current_stage)]
            val_score = float(val_summary["mean_task_score"])
            val_model_score = _validation_model_score(val_summary)
            train_mean_task_score = (
                float(np.mean(rolling_task_scores)) if rolling_task_scores else task_score
            )
            is_best_val = (
                bool(config["save_best_by_validation"])
                and curriculum.current_stage == 3
                and curriculum.stage_episodes[3] >= 300  # 阶段3至少运行300回合
                and curriculum._terminal_focus_active     # 终末专注已激活
                and val_model_score > best_val_model_score
            )
            if is_best_val:
                best_val_model_score = val_model_score
                best_val_path = os.path.join(model_dir, "ppo_best_val.pth")
                agent.save(best_val_path)
                best_model_paths["best_val"] = best_val_path
                print(
                    f"  -> 最佳验证模型分数: {best_val_model_score * 100:.1f}% "
                    f"(任务得分={val_score * 100:.1f}%)"
                )

            validation_log["episodes"].append(episode)
            validation_log["stage"].append(curriculum.current_stage)
            validation_log["train_task_score"].append(train_mean_task_score)
            validation_log["val_mean_task_score"].append(val_score)
            validation_log["val_mean_coverage"].append(float(val_summary["mean_coverage"]))
            validation_log["val_success_rate"].append(float(val_summary["success_rate"]))
            validation_log["val_mean_length"].append(float(val_summary["mean_length"]))
            validation_log["val_timeout_rate"].append(float(val_summary["timeout_rate"]))
            validation_log["val_zero_coverage_timeout_rate"].append(
                float(val_summary["zero_coverage_timeout_rate"])
            )
            validation_log["generalization_gap"].append(train_mean_task_score - val_score)
            validation_log["is_best_val"].append(1 if is_best_val else 0)

        if episode % config["save_interval"] == 0:
            checkpoint_path = os.path.join(model_dir, f"ppo_ep{episode}_stage{curriculum.current_stage}.pth")
            agent.save(checkpoint_path)
            mean_task_score = float(np.mean(rolling_task_scores)) if rolling_task_scores else task_score
            if mean_task_score > best_task_score:
                best_task_score = mean_task_score
                best_path = os.path.join(model_dir, "ppo_best_train.pth")
                agent.save(best_path)
                legacy_best_path = os.path.join(model_dir, "ppo_best.pth")
                agent.save(legacy_best_path)
                best_model_paths["best"] = best_path
                best_model_paths["best_train"] = best_path
                print(f"  -> 最佳训练滚动任务得分: {best_task_score * 100:.1f}%")

        max_train_updates = config.get("max_train_updates")
        if max_train_updates is not None and agent.training_step >= max_train_updates:
            print(f"达到 PPO 更新预算: {agent.training_step}/{max_train_updates}")
            break

    if len(agent.buffer) >= agent.min_update_batch_size:
        update_info = agent.update(force=True)

    final_model_path = os.path.join(model_dir, "ppo_final.pth")
    agent.save(final_model_path)
    best_model_paths["final"] = final_model_path

    log_path = os.path.join(log_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(training_log), f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(log_dir, "training_log.npz"), **training_log)

    validation_log_path = os.path.join(log_dir, "validation_log.json")
    with open(validation_log_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(validation_log), f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(log_dir, "validation_log.npz"), **validation_log)

    quality_metrics = compute_model_quality_metrics(training_log, config)
    metrics_path = os.path.join(log_dir, "model_quality_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(quality_metrics), f, indent=2, ensure_ascii=False)

    generalization_path = None
    eval_results_path = None
    eval_summary_path = os.path.join(log_dir, "eval_summary.json")
    eval_summary = {
        "best_val": {
            "available": bool(best_model_paths.get("best_val")),
            "model_path": best_model_paths.get("best_val"),
            "splits": {},
        },
    }
    if config["eval_after_train"]:
        print("\n" + "=" * 70)
        print("训练后最佳验证模型评估")
        print(
            f"数据划分={config['final_eval_splits']} | 评估阶段={config['eval_stages']} | "
            f"每场景回合={config['final_eval_episodes_per_scene']}"
        )
        print("=" * 70)

        eval_checkpoints = _after_train_eval_checkpoints(config, best_model_paths)
        if not eval_checkpoints:
            print("未找到 best-val checkpoint，跳过训练后评估")

        for checkpoint_name, checkpoint_path in eval_checkpoints:
            best_val_agent = CTDE_PPO_Agent(
                local_obs_dim=env.local_obs_dim,
                global_state_dim=env.global_state_dim,
                action_dim=env.num_actions,
                num_agents=env.num_drones,
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                lr_adapt_mode=config["lr_adapt_mode"],
                target_kl=config["target_kl"],
                actor_lr_min=config["actor_lr_min"],
                actor_lr_max=config["actor_lr_max"],
                kl_ema_beta=config["kl_ema_beta"],
                kl_lr_alpha=config["kl_lr_alpha"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_epsilon=config["clip_epsilon"],
                entropy_coef=config["entropy_coef"],
                value_coef=config["value_coef"],
                max_grad_norm=config["max_grad_norm"],
                ppo_epochs=config["ppo_epochs"],
                batch_size=config["batch_size"],
            )
            best_val_agent.load(checkpoint_path)

            for split in config["final_eval_splits"]:
                split_config = make_eval_config(
                    config,
                    split,
                    config["final_eval_episodes_per_scene"],
                    config["eval_stages"],
                )
                print(f"\n{checkpoint_name}模型评估 | 数据划分={split}")
                split_results = evaluate(best_val_agent, split_config)
                split_path = os.path.join(log_dir, f"{checkpoint_name}_{split}_eval.json")
                _write_json(split_path, split_results)
                eval_summary[checkpoint_name]["splits"][split] = {
                    "path": split_path,
                    "stages": {
                        str(stage): _stage_summary(split_results, int(stage))
                        for stage in config["eval_stages"]
                    },
                }
                if checkpoint_name == "best_val" and split == "generalization":
                    generalization_path = os.path.join(log_dir, "generalization_eval.json")
                    eval_results_path = os.path.join(log_dir, "eval_results.json")
                    _write_json(generalization_path, split_results)
                    _write_json(eval_results_path, split_results)
                    print(f"generalization_eval={generalization_path}")
                    print(f"eval_results={eval_results_path}")
                print(f"{checkpoint_name}_{split}_eval={split_path}")

        _write_json(eval_summary_path, eval_summary)
        print(f"eval_summary={eval_summary_path}")

    source_dir = _save_source_snapshot(output_dir)
    figure_paths = {}
    if config["plot_after_train"]:
        figure_paths = _run_figure_scripts(
            output_dir,
            config,
            include_generalization=bool(config["eval_after_train"] and generalization_path),
        )

    elapsed_min = (time.time() - start_time) / 60.0
    print("\n" + "=" * 70)
    print("训练完成")
    print(f"总步数={total_steps} | PPO更新={agent.training_step} | 用时={elapsed_min:.1f} 分钟")
    print(f"最终阶段={curriculum.current_stage}")
    print(f"训练日志={log_path}")
    print(f"验证日志={validation_log_path}")
    print(f"质量指标={metrics_path}")
    if config["eval_after_train"]:
        if generalization_path:
            print(f"泛化评估={generalization_path}")
        if eval_results_path:
            print(f"评估结果={eval_results_path}")
        print(f"评估摘要={eval_summary_path}")
    print(f"最终模型={final_model_path}")
    print(f"控制台日志={console_log_path}")
    print(f"源码快照={source_dir}")
    if figure_paths:
        print(f"图表={figure_paths}")
    print(
        "质量摘要 | "
        f"AUC={quality_metrics.get('convergence_efficiency', {}).get('auc_task_score_by_steps')} | "
        f"tail得分标准差={quality_metrics.get('reward_stability', {}).get('task_score_std_tail')} | "
        f"KL超限率={quality_metrics.get('kl_stability', {}).get('kl_overshoot_rate')}"
    )
    print("=" * 70)

    return agent, curriculum, training_log, output_dir, best_model_paths


def evaluate(agent: CTDE_PPO_Agent, config: Dict, num_episodes: int = None) -> Dict:
    config = normalize_training_config(config)
    _resolve_dataset_scene_keys(config)
    eval_episodes_per_scene = int(config["eval_episodes_per_scene"])
    if num_episodes is not None:
        eval_episodes_per_scene = max(1, int(num_episodes) // max(len(config["eval_scene_keys"]), 1))

    actor_was_training = agent.actor.training
    critic_was_training = agent.critic.training
    agent.actor.eval()
    agent.critic.eval()

    results = {}
    base_seed = config["seed"]

    try:
        for stage in config["eval_stages"]:
            stage_records = []
            stage_seed = base_seed + int(stage) * config["eval_seed_stride"]
            np.random.seed(stage_seed)
            random.seed(stage_seed)

            for scene_key in config["eval_scene_keys"]:
                env = FireSearchBaselineEnvironment(
                    data_dir=config["data_dir"],
                    num_drones=config["num_drones"],
                    vision_radius=config["vision_radius"],
                    max_steps=config["max_steps"],
                    use_metadata_uav_params=config["use_metadata_uav_params"],
                    observation_profile=config["observation_profile"],
                    reward_profile=config["reward_profile"],
                    curriculum_stage=int(stage),
                    mode=config["eval_split"],
                    fixed_scene_key=str(scene_key),
                    init_percentile=config["init_percentile"],
                    init_area_percent=config["init_area_percent"],
                    stage2_target=config["stage2_success_target"],
                    stage3_target=config["stage3_success_target"],
                    stage3_near_prob=0.0,
                )

                for _ in range(eval_episodes_per_scene):
                    obs = env.reset()
                    done = False
                    episode_reward = 0.0
                    episode_length = 0
                    while not done:
                        actions = agent.select_actions_deterministic(obs["local_obs"])
                        obs, rewards, done, info = env.step(actions)
                        episode_reward += float(sum(rewards))
                        episode_length += 1

                    success = info["done_reason"] == "mission_complete"
                    stage_records.append(
                        {
                            "scene_id": int(info.get("scene_id", -1)),
                            "scene_key": str(info.get("scene_key", scene_key)),
                            "observation_profile": config["observation_profile"],
                            "reward_profile": config["reward_profile"],
                            "reward": episode_reward,
                            "coverage": float(info["boundary_coverage"]),
                            "success": 1.0 if success else 0.0,
                            "length": int(episode_length),
                            "timeout": 1.0 if info["done_reason"] == "max_steps_reached" else 0.0,
                            "zero_coverage_timeout": 1.0 if info.get("zero_coverage_timeout", False) else 0.0,
                            "task_score": _task_score(
                                info["boundary_coverage"],
                                success,
                                episode_length,
                                config["max_steps"],
                            ),
                            "done_reason": info["done_reason"],
                            "first_heat_step": int(info.get("first_heat_step", -1)),
                            "first_boundary_step": int(info.get("first_boundary_step", -1)),
                            "reward_breakdown": info.get("reward_breakdown") or {},
                        }
                    )

            results[int(stage)] = {
                "episodes": len(stage_records),
                "mean_reward": float(np.mean([r["reward"] for r in stage_records])),
                "mean_coverage": float(np.mean([r["coverage"] for r in stage_records])),
                "success_rate": float(np.mean([r["success"] for r in stage_records])),
                "mean_length": float(np.mean([r["length"] for r in stage_records])),
                "timeout_rate": float(np.mean([r["timeout"] for r in stage_records])),
                "zero_coverage_timeout_rate": float(
                    np.mean([r["zero_coverage_timeout"] for r in stage_records])
                ),
                "mean_task_score": float(np.mean([r["task_score"] for r in stage_records])),
                "records": stage_records,
            }

            summary = results[int(stage)]
            print(
                f"评估阶段={stage} | 回合数={summary['episodes']} | "
                f"任务得分={summary['mean_task_score'] * 100:.1f}% | "
                f"覆盖率={summary['mean_coverage'] * 100:.1f}% | "
                f"成功率={summary['success_rate'] * 100:.1f}% | "
                f"平均步数={summary['mean_length']:.1f}"
            )
    finally:
        agent.actor.train(actor_was_training)
        agent.critic.train(critic_was_training)

    return results


def run_lr_comparison(base_config: Dict = None) -> Dict:
    base_config = normalize_training_config(base_config)

    output_root = base_config.get("output_root_dir", "./outputs")
    if not os.path.isabs(output_root):
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = os.path.join(os.path.abspath(output_root), f"lr_comparison_{timestamp}")
    results_root = os.path.join(package_dir, RESULTS_DIR_NAME)
    os.makedirs(results_root, exist_ok=True)

    variants = [
        (
            "Fixed_LR_CTDE_PPO",
            {
                "lr_adapt_mode": "fixed",
                "actor_lr": base_config["actor_lr"],
            },
        ),
        (
            "KL_LR_CTDE_PPO",
            {
                "lr_adapt_mode": "kl",
                "actor_lr": base_config["actor_lr"],
                "target_kl": base_config["target_kl"],
                "quality_target_kl": base_config["target_kl"],
            },
        ),
    ]

    comparison_results = {
        "package_dir": package_dir,
        "results_root": results_root,
        "seeds": list(base_config["comparison_seeds"]),
        "variants": {},
    }

    for seed in base_config["comparison_seeds"]:
        for variant_name, overrides in variants:
            cfg = copy.deepcopy(base_config)
            cfg.update(overrides)
            cfg["seed"] = int(seed)
            cfg["variant_name"] = variant_name
            cfg["output_dir"] = os.path.join(results_root, f"{variant_name}_seed{cfg['seed']}")
            cfg["plot_after_train"] = False

            print("\n" + "=" * 80)
            print(f"开始学习率对比变体: {variant_name} | 随机种子={cfg['seed']}")
            print(f"输出目录={cfg['output_dir']}")
            print("=" * 80)

            trained_agent, curriculum, training_log, output_dir, best_model_paths = train(cfg)
            del trained_agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metrics_path = os.path.join(output_dir, "logs", "model_quality_metrics.json")
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

            generalization_path = os.path.join(output_dir, "logs", "generalization_eval.json")
            if not os.path.exists(generalization_path):
                generalization_path = os.path.join(
                    output_dir,
                    "logs",
                    "best_val_generalization_eval.json",
                )
            generalization = {}
            if os.path.exists(generalization_path):
                with open(generalization_path, "r", encoding="utf-8") as f:
                    generalization = json.load(f)

            eval_summary_path = os.path.join(output_dir, "logs", "eval_summary.json")
            eval_summary = {}
            if os.path.exists(eval_summary_path):
                with open(eval_summary_path, "r", encoding="utf-8") as f:
                    eval_summary = json.load(f)

            run_name = f"{variant_name}_seed{cfg['seed']}"
            comparison_results["variants"][run_name] = {
                "config": _to_jsonable(cfg),
                "variant_name": variant_name,
                "seed": int(seed),
                "output_dir": output_dir,
                "final_stage": curriculum.current_stage,
                "best_model_paths": _to_jsonable(best_model_paths),
                "quality_metrics_path": metrics_path,
                "quality_metrics": metrics,
                "generalization_eval_path": generalization_path,
                "generalization_eval": generalization,
                "eval_summary_path": eval_summary_path,
                "eval_summary": eval_summary,
                "last_task_score": float(training_log["task_scores"][-1]) if training_log["task_scores"] else None,
                "last_coverage": float(training_log["coverages"][-1]) if training_log["coverages"] else None,
                "ppo_updates": int(training_log["ppo_updates"][-1]) if training_log["ppo_updates"] else 0,
            }

    summary_path = os.path.join(package_dir, "lr_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(comparison_results), f, indent=2, ensure_ascii=False)

    figure_paths = {}
    if base_config["plot_after_train"]:
        figures_root = os.path.join(package_dir, "figures")
        for seed in base_config["comparison_seeds"]:
            seed_key = f"seed{int(seed)}"
            figure_paths[seed_key] = _run_figure_scripts(
                results_root,
                base_config,
                include_generalization=bool(base_config["eval_after_train"]),
                out_root=os.path.join(figures_root, seed_key),
                seed_filter=int(seed),
            )
        figure_paths["summary_curves"] = _run_figure_scripts(
            results_root,
            base_config,
            include_generalization=bool(base_config["eval_after_train"]),
            out_root=os.path.join(figures_root, "summary_curves"),
            aggregate_seeds=True,
        )
        comparison_results["figure_paths"] = figure_paths
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(comparison_results), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("学习率对比完成")
    print(f"汇总文件={summary_path}")
    if figure_paths:
        print(f"图表={figure_paths}")
    for run_name, result in comparison_results["variants"].items():
        metrics = result.get("quality_metrics", {})
        conv = metrics.get("convergence_efficiency", {})
        stable = metrics.get("reward_stability", {})
        kl = metrics.get("kl_stability", {})
        generalization = result.get("generalization_eval", {})
        stage_key = str(base_config["eval_stages"][0]) if base_config["eval_stages"] else "3"
        gen_stage = generalization.get(stage_key, {})
        eval_summary = result.get("eval_summary", {})
        if not gen_stage:
            gen_stage = _eval_summary_stage(eval_summary, "generalization", stage_key)
        stress_stage = _eval_summary_stage(eval_summary, "stress", stage_key)
        print(
            f"{run_name}: "
            f"AUC={conv.get('auc_task_score_by_steps')} | "
            f"tail得分标准差={stable.get('task_score_std_tail')} | "
            f"KL超限率={kl.get('kl_overshoot_rate')} | "
            f"泛化任务得分={gen_stage.get('mean_task_score')} | "
            f"泛化成功率={gen_stage.get('success_rate')} | "
            f"压力测试任务得分={stress_stage.get('mean_task_score')} | "
            f"输出目录={result['output_dir']}"
        )
    print("=" * 80)

    return comparison_results


def _prompt_seed_mode() -> str:
    while True:
        choice = input("Select seed mode: 1 = single seed 42, 3 = seeds 42/43/44: ").strip()
        if choice == "1":
            return "single"
        if choice == "3":
            return "three"
        print("Please enter 1 or 3.")


def main():
    parser = argparse.ArgumentParser(description="Train clean baseline CTDE-PPO.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--eval-scene-keys", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr-comparison", action="store_true")
    parser.add_argument("--single-train", action="store_true")
    parser.add_argument("--lr-mode", choices=["fixed", "kl"], default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--kl-lr-alpha", type=float, default=None)
    parser.add_argument("--init-percentile", type=float, default=None)
    parser.add_argument("--init-area-percent", type=float, default=None)
    parser.add_argument("--use-metadata-uav-params", action="store_true")
    parser.add_argument("--observation-profile", type=str, default=None)
    parser.add_argument("--reward-profile", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = {}
    if args.episodes is not None:
        config["total_episodes"] = args.episodes
    if args.max_updates is not None:
        config["max_train_updates"] = args.max_updates
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.train_split is not None:
        config["train_split"] = args.train_split
    if args.eval_split is not None:
        config["eval_split"] = args.eval_split
    if args.eval_scene_keys is not None:
        config["eval_scene_keys"] = [item.strip() for item in args.eval_scene_keys.split(",") if item.strip()]
    if args.seed is not None:
        config["seed"] = args.seed
    if not args.single_train:
        seed_mode = _prompt_seed_mode()
        config["comparison_seeds"] = [42] if seed_mode == "single" else [42, 43, 44]
    if args.lr_mode is not None:
        config["lr_adapt_mode"] = args.lr_mode
    if args.target_kl is not None:
        config["target_kl"] = args.target_kl
        config["quality_target_kl"] = args.target_kl
    if args.kl_lr_alpha is not None:
        config["kl_lr_alpha"] = args.kl_lr_alpha
    if args.init_percentile is not None:
        config["init_percentile"] = args.init_percentile
        config["init_area_percent"] = args.init_percentile
    if args.init_area_percent is not None:
        config["init_area_percent"] = args.init_area_percent
        config["init_percentile"] = args.init_area_percent
    if args.use_metadata_uav_params:
        config["use_metadata_uav_params"] = True
    if args.observation_profile is not None:
        config["observation_profile"] = args.observation_profile
    if args.reward_profile is not None:
        config["reward_profile"] = args.reward_profile
    if args.no_eval:
        config["eval_after_train"] = False
    if args.no_plot:
        config["plot_after_train"] = False

    if args.single_train:
        train(config)
    else:
        run_lr_comparison(config)


if __name__ == "__main__":
    main()
