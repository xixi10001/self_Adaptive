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
    "observation_profile": "persistent_cooperative",
    "reward_profile": "persistent_boundary",
    "communication_enabled": True,
    "communication_radius_factor": 4.0,
    "action_mask_enabled": True,
    "novelty_reward_weight": 0.12,
    "novelty_step_penalty": -0.04,
    "novelty_revisit_penalty": 0.08,
    "invalid_action_penalty": 0.25,
    "team_overlap_penalty": 0.05,
    "hierarchical_roles_enabled": True,
    "peer_state_ttl": 8,
    "track_report_ttl": 20,
    "reacquire_report_ttl": 60,
    "role_decision_interval": 20,
    "role_min_dwell_steps": 10,
    "boundary_match_radius": 2,
    "boundary_freshness_tau": 40.0,
    "fresh_coverage_gain_weight": 20.0,
    "assigned_boundary_gain_weight": 1.0,
    "role_switch_penalty": 0.05,
    "mask_thermal_below_signal": True,
    "role_actor_lr": 2e-4,
    "role_critic_lr": 5e-4,
    "role_batch_size": 128,
    "norm_params_source": "scene_p99.5",
    "init_percentile": 5.0,
    "init_area_percent": 5.0,
    "total_episodes": 3100,
    "max_environment_steps": None,
    "max_train_updates": None,
    "actor_lr": 2e-4,
    "critic_lr": 5e-4,
    "lr_adapt_mode": "fixed",
    "target_kl": 0.0065,
    "actor_lr_min": 1e-4,
    "actor_lr_max": 2.5e-4,
    "kl_ema_beta": 0.8,
    "kl_lr_low_ratio": 0.82,
    "kl_lr_high_ratio": 1.20,
    "kl_lr_emergency_ratio": 2.00,
    "kl_lr_up_factor": 1.03,
    "kl_lr_down_factor": 0.90,
    "kl_lr_emergency_factor": 0.70,
    "kl_lr_low_patience": 3,
    "kl_early_stop_ratio": 1.50,
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
    "stage2_success_target": 0.20,
    "stage3_success_target": 0.60,
    "stage3_near_prob": 0.0,
    "validation_split": "validation",
    "validation_interval": 50,
    "validation_episodes_per_scene": 5,
    "save_best_by_validation": True,
    "eval_scene_keys": None,
    "eval_episodes_per_scene": 50,
    "eval_stages": [3],
    "eval_seed_stride": 100,
    "evaluation_mode": "target_stop",
    "eval_after_train": True,
    "final_eval_splits": ["validation", "generalization", "stress"],
    "final_eval_episodes_per_scene": 50,
    "evaluate_best_val_after_train": True,
    "full_horizon_eval_after_train": True,
    "full_horizon_eval_splits": ["validation", "generalization", "stress"],
    "full_horizon_validation_episodes_per_scene": 20,
    "full_horizon_final_episodes_per_scene": 50,
    "full_horizon_curve_stride": 20,
    "full_horizon_thresholds": [0.20, 0.40, 0.60, 0.80],
    "post_target_train": False,
    "post_target_resume_checkpoint": None,
    "post_target_episodes": 500,
    "post_target_goal_ladder": [0.60, 0.65, 0.70],
    "post_target_validation_patience": 2,
    "post_target_step_penalty": -0.02,
    "post_target_step_cost_fraction": 0.25,
    "post_target_hold_weight": 0.10,
    "post_target_tail_weight": 20.0,
    "post_target_milestone_70": 5.0,
    "post_target_milestone_80": 10.0,
    "post_target_actor_lr": 1e-4,
    "post_target_critic_lr": 2.5e-4,
    "post_target_warmup_updates": 15,
    "post_target_initial_clip_epsilon": 0.15,
    "post_target_initial_max_grad_norm": 0.25,
    "quality_score_threshold": 0.55,
    "quality_window": 50,
    "quality_tail_fraction": 0.2,
    "quality_target_kl": 0.0065,
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
    if "communication_enabled" not in user_keys:
        normalized["communication_enabled"] = (
            normalized["observation_profile"] == "cooperative_exploration"
        )
    if "action_mask_enabled" not in user_keys:
        normalized["action_mask_enabled"] = (
            normalized["observation_profile"] == "cooperative_exploration"
            or normalized["reward_profile"] == "novelty_search"
        )
    normalized["communication_enabled"] = bool(normalized["communication_enabled"])
    normalized["communication_radius_factor"] = max(
        0.0, float(normalized["communication_radius_factor"])
    )
    normalized["action_mask_enabled"] = bool(normalized["action_mask_enabled"])
    normalized["novelty_reward_weight"] = max(
        0.0, float(normalized["novelty_reward_weight"])
    )
    normalized["novelty_step_penalty"] = float(normalized["novelty_step_penalty"])
    normalized["novelty_revisit_penalty"] = max(
        0.0, float(normalized["novelty_revisit_penalty"])
    )
    normalized["invalid_action_penalty"] = max(
        0.0, float(normalized["invalid_action_penalty"])
    )
    normalized["team_overlap_penalty"] = max(
        0.0, float(normalized["team_overlap_penalty"])
    )
    if "hierarchical_roles_enabled" not in user_keys:
        normalized["hierarchical_roles_enabled"] = (
            normalized["observation_profile"] == "persistent_cooperative"
        )
    if normalized["observation_profile"] == "persistent_cooperative":
        normalized["hierarchical_roles_enabled"] = True
        normalized["communication_enabled"] = True
        normalized["action_mask_enabled"] = True
        normalized["mask_thermal_below_signal"] = True
    if normalized["reward_profile"] == "persistent_boundary":
        normalized["observation_profile"] = "persistent_cooperative"
        normalized["hierarchical_roles_enabled"] = True
        normalized["communication_enabled"] = True
        normalized["action_mask_enabled"] = True
        normalized["mask_thermal_below_signal"] = True
    normalized["hierarchical_roles_enabled"] = bool(
        normalized["hierarchical_roles_enabled"]
    )
    if normalized["hierarchical_roles_enabled"] and int(normalized["num_drones"]) != 2:
        raise ValueError("hierarchical_roles_enabled currently requires num_drones=2")
    normalized["peer_state_ttl"] = max(1, int(normalized["peer_state_ttl"]))
    normalized["track_report_ttl"] = max(1, int(normalized["track_report_ttl"]))
    normalized["reacquire_report_ttl"] = max(
        normalized["track_report_ttl"] + 1,
        int(normalized["reacquire_report_ttl"]),
    )
    normalized["role_decision_interval"] = max(
        1, int(normalized["role_decision_interval"])
    )
    normalized["role_min_dwell_steps"] = max(
        0, int(normalized["role_min_dwell_steps"])
    )
    normalized["boundary_match_radius"] = max(
        0, int(normalized["boundary_match_radius"])
    )
    normalized["boundary_freshness_tau"] = max(
        1.0, float(normalized["boundary_freshness_tau"])
    )
    normalized["fresh_coverage_gain_weight"] = max(
        0.0, float(normalized["fresh_coverage_gain_weight"])
    )
    normalized["assigned_boundary_gain_weight"] = max(
        0.0, float(normalized["assigned_boundary_gain_weight"])
    )
    normalized["role_switch_penalty"] = max(
        0.0, float(normalized["role_switch_penalty"])
    )
    normalized["mask_thermal_below_signal"] = bool(
        normalized["mask_thermal_below_signal"]
    )
    normalized["role_actor_lr"] = float(normalized["role_actor_lr"])
    normalized["role_critic_lr"] = float(normalized["role_critic_lr"])
    normalized["role_batch_size"] = max(8, int(normalized["role_batch_size"]))
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
    if normalized.get("max_environment_steps") is not None:
        normalized["max_environment_steps"] = max(
            1, int(normalized["max_environment_steps"])
        )
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
    normalized["kl_lr_low_ratio"] = max(0.0, float(normalized["kl_lr_low_ratio"]))
    normalized["kl_lr_high_ratio"] = max(
        normalized["kl_lr_low_ratio"], float(normalized["kl_lr_high_ratio"])
    )
    normalized["kl_lr_emergency_ratio"] = max(
        normalized["kl_lr_high_ratio"], float(normalized["kl_lr_emergency_ratio"])
    )
    normalized["kl_lr_up_factor"] = max(1.0, float(normalized["kl_lr_up_factor"]))
    normalized["kl_lr_down_factor"] = float(np.clip(normalized["kl_lr_down_factor"], 1e-6, 1.0))
    normalized["kl_lr_emergency_factor"] = float(
        np.clip(normalized["kl_lr_emergency_factor"], 1e-6, normalized["kl_lr_down_factor"])
    )
    normalized["kl_lr_low_patience"] = max(1, int(normalized["kl_lr_low_patience"]))
    normalized["kl_early_stop_ratio"] = max(1.0, float(normalized["kl_early_stop_ratio"]))
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
    normalized["evaluation_mode"] = str(normalized.get("evaluation_mode", "target_stop")).lower()
    if normalized["evaluation_mode"] not in FireSearchBaselineEnvironment.TERMINATION_MODES:
        raise ValueError("evaluation_mode must be 'target_stop' or 'full_horizon'")
    normalized["eval_after_train"] = bool(normalized["eval_after_train"])
    normalized["final_eval_splits"] = _normalize_str_list(normalized.get("final_eval_splits", ["validation", "generalization", "stress"]))
    normalized["final_eval_episodes_per_scene"] = max(1, int(normalized["final_eval_episodes_per_scene"]))
    normalized["evaluate_best_val_after_train"] = bool(normalized["evaluate_best_val_after_train"])
    normalized["full_horizon_eval_after_train"] = bool(normalized["full_horizon_eval_after_train"])
    normalized["full_horizon_eval_splits"] = _normalize_str_list(
        normalized.get("full_horizon_eval_splits", ["validation", "generalization", "stress"])
    )
    normalized["full_horizon_validation_episodes_per_scene"] = max(
        1, int(normalized["full_horizon_validation_episodes_per_scene"])
    )
    normalized["full_horizon_final_episodes_per_scene"] = max(
        1, int(normalized["full_horizon_final_episodes_per_scene"])
    )
    normalized["full_horizon_curve_stride"] = max(1, int(normalized["full_horizon_curve_stride"]))
    normalized["full_horizon_thresholds"] = sorted(
        {
            float(np.clip(value, 0.0, 1.0))
            for value in normalized.get("full_horizon_thresholds", [0.20, 0.40, 0.60, 0.80])
        }
    )
    normalized["post_target_train"] = bool(normalized["post_target_train"])
    normalized["post_target_episodes"] = max(1, int(normalized["post_target_episodes"]))
    normalized["post_target_goal_ladder"] = sorted(
        {
            float(np.clip(value, 0.60, 0.80))
            for value in normalized.get("post_target_goal_ladder", [0.60, 0.65, 0.70])
        }
    )
    if not normalized["post_target_goal_ladder"]:
        raise ValueError("post_target_goal_ladder must not be empty")
    normalized["post_target_validation_patience"] = max(
        1, int(normalized["post_target_validation_patience"])
    )
    for key in [
        "post_target_step_penalty",
        "post_target_step_cost_fraction",
        "post_target_hold_weight",
        "post_target_tail_weight",
        "post_target_milestone_70",
        "post_target_milestone_80",
        "post_target_actor_lr",
        "post_target_critic_lr",
        "post_target_initial_clip_epsilon",
        "post_target_initial_max_grad_norm",
    ]:
        normalized[key] = float(normalized[key])
    normalized["post_target_step_cost_fraction"] = max(
        0.0, normalized["post_target_step_cost_fraction"]
    )
    normalized["post_target_warmup_updates"] = max(
        0, int(normalized["post_target_warmup_updates"])
    )
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
    training_log.setdefault("communication_available_rate", []).append(
        float(info.get("communication_available_rate", 0.0))
    )
    training_log.setdefault("shared_new_cells", []).append(int(info.get("shared_new_cells", 0)))
    training_log.setdefault("pre_boundary_revisit_ratio", []).append(
        float(info.get("pre_boundary_revisit_ratio", 0.0))
    )
    training_log.setdefault("team_overlap_ratio", []).append(
        float(info.get("team_overlap_ratio", 0.0))
    )
    training_log.setdefault("invalid_action_count", []).append(
        int(info.get("invalid_action_count", 0))
    )
    for key in [
        "objective_coverage",
        "fresh_boundary_coverage",
        "tolerant_boundary_coverage",
        "communication_message_expirations",
        "boundary_report_expirations",
        "role_switch_count",
        "task_conflict_count",
        "zero_discovery_timeout",
        "stage1_tracking_success",
        "stage1_tracking_steps",
        "stage1_unique_boundary_cells",
        "major_refresh_count",
        "refresh_recovery_successes",
        "mean_refresh_recovery_time",
    ]:
        training_log.setdefault(key, []).append(info.get(key, 0))
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

    def get_action_probs(
        self, local_obs: torch.Tensor, action_masks: torch.Tensor = None
    ) -> Categorical:
        logits = self.forward(local_obs)
        if action_masks is not None:
            masks = action_masks.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)


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
        self.next_global_states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.modes = []

    def store(
        self,
        local_obs,
        global_state,
        actions,
        log_probs,
        rewards,
        done,
        next_global_state=None,
        truncated=False,
        mode=0,
        action_masks=None,
    ):
        self.local_obs.append(local_obs)
        self.global_states.append(global_state)
        self.next_global_states.append(
            global_state if next_global_state is None else next_global_state
        )
        self.actions.append(actions)
        self.action_masks.append(action_masks)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.terminated.append(bool(done and not truncated))
        self.truncated.append(bool(truncated))
        self.modes.append(int(mode))

    def clear(self):
        self.local_obs = []
        self.global_states = []
        self.next_global_states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.modes = []

    def get(self):
        return (
            self.local_obs,
            self.global_states,
            self.next_global_states,
            self.actions,
            self.action_masks,
            self.log_probs,
            self.rewards,
            self.terminated,
            self.truncated,
            self.modes,
        )

    def __len__(self):
        return len(self.rewards)


class RolePolicyNetwork(nn.Module):
    def __init__(self, role_obs_dim: int, num_roles: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(role_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_roles),
        )

    def forward(self, role_obs: torch.Tensor) -> torch.Tensor:
        return self.network(role_obs)


class RoleCriticNetwork(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state)


class OptionRolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.role_obs = []
        self.global_states = []
        self.next_global_states = []
        self.joint_masks = []
        self.joint_actions = []
        self.log_probs = []
        self.option_rewards = []
        self.durations = []
        self.terminated = []
        self.truncated = []

    def store(
        self,
        role_obs,
        global_state,
        next_global_state,
        joint_mask,
        joint_action,
        log_prob,
        option_reward,
        duration,
        terminated,
        truncated,
    ):
        self.role_obs.append(role_obs)
        self.global_states.append(global_state)
        self.next_global_states.append(next_global_state)
        self.joint_masks.append(joint_mask)
        self.joint_actions.append(int(joint_action))
        self.log_probs.append(float(log_prob))
        self.option_rewards.append(float(option_reward))
        self.durations.append(max(1, int(duration)))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def __len__(self):
        return len(self.option_rewards)


class RolePPOAgent:
    """SMDP PPO over constrained joint roles for exactly two agents."""

    def __init__(
        self,
        role_obs_dim: int,
        global_state_dim: int,
        num_roles: int,
        device: torch.device,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        entropy_coef: float,
        value_coef: float,
        max_grad_norm: float,
        ppo_epochs: int,
        batch_size: int,
    ):
        self.role_obs_dim = int(role_obs_dim)
        self.global_state_dim = int(global_state_dim)
        self.num_roles = int(num_roles)
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = max(8, int(batch_size))
        self.actor = RolePolicyNetwork(role_obs_dim, num_roles).to(device)
        self.critic = RoleCriticNetwork(global_state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = OptionRolloutBuffer()
        self._active_option = None

    def _joint_distribution(
        self, role_obs: torch.Tensor, joint_masks: torch.Tensor
    ) -> Categorical:
        batch_shape = role_obs.shape[:-2]
        flat_obs = role_obs.reshape(-1, role_obs.shape[-1])
        local_logits = self.actor(flat_obs).reshape(
            *batch_shape, 2, self.num_roles
        )
        joint_logits = (
            local_logits[..., 0, :, None] + local_logits[..., 1, None, :]
        ).reshape(*batch_shape, self.num_roles**2)
        masks = joint_masks.to(device=joint_logits.device, dtype=torch.bool)
        joint_logits = joint_logits.masked_fill(
            ~masks, torch.finfo(joint_logits.dtype).min
        )
        return Categorical(logits=joint_logits)

    def select_joint_roles(
        self,
        role_obs: List[np.ndarray],
        joint_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[List[int], float, int]:
        obs_tensor = torch.as_tensor(
            np.asarray(role_obs), dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(
            np.asarray(joint_mask), dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            distribution = self._joint_distribution(obs_tensor, mask_tensor)
            joint_action = (
                torch.argmax(distribution.logits)
                if deterministic
                else distribution.sample()
            )
            log_prob = distribution.log_prob(joint_action)
        action_idx = int(joint_action.item())
        return (
            [action_idx // self.num_roles, action_idx % self.num_roles],
            float(log_prob.item()),
            action_idx,
        )

    def imitate_joint_roles(
        self,
        role_obs: List[np.ndarray],
        joint_mask: np.ndarray,
        roles: List[int],
    ) -> float:
        obs_tensor = torch.as_tensor(
            np.asarray(role_obs), dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(
            np.asarray(joint_mask), dtype=torch.bool, device=self.device
        )
        joint_action = torch.tensor(
            int(roles[0]) * self.num_roles + int(roles[1]),
            dtype=torch.long,
            device=self.device,
        )
        distribution = self._joint_distribution(obs_tensor, mask_tensor)
        loss = -distribution.log_prob(joint_action)
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        return float(loss.item())

    def begin_option(
        self,
        role_obs,
        global_state,
        joint_mask,
        joint_action,
        log_prob,
    ) -> None:
        if self._active_option is not None:
            raise RuntimeError("cannot begin a role option before finishing the active option")
        self._active_option = {
            "role_obs": np.asarray(role_obs, dtype=np.float32),
            "global_state": np.asarray(global_state, dtype=np.float32),
            "joint_mask": np.asarray(joint_mask, dtype=np.int8),
            "joint_action": int(joint_action),
            "log_prob": float(log_prob),
            "reward": 0.0,
            "duration": 0,
        }

    def accumulate_reward(self, team_reward: float) -> None:
        if self._active_option is None:
            return
        duration = int(self._active_option["duration"])
        self._active_option["reward"] += (self.gamma**duration) * float(team_reward)
        self._active_option["duration"] = duration + 1

    def finish_option(
        self,
        next_global_state,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._active_option is None:
            return
        option = self._active_option
        if int(option["duration"]) > 0:
            self.buffer.store(
                option["role_obs"],
                option["global_state"],
                np.asarray(next_global_state, dtype=np.float32),
                option["joint_mask"],
                option["joint_action"],
                option["log_prob"],
                option["reward"],
                option["duration"],
                terminated,
                truncated,
            )
        self._active_option = None

    def compute_smdp_gae(
        self,
        rewards: torch.Tensor,
        durations: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
        for idx in reversed(range(rewards.shape[0])):
            duration = float(durations[idx].item())
            gamma_d = self.gamma**duration
            trace_d = (self.gamma * self.gae_lambda) ** duration
            bootstrap = 0.0 if terminated[idx] else 1.0
            trace = 0.0 if (terminated[idx] or truncated[idx]) else 1.0
            delta = (
                rewards[idx]
                + gamma_d * bootstrap * next_values[idx]
                - values[idx]
            )
            gae = delta + trace_d * trace * gae
            advantages[idx] = gae
        return advantages, advantages + values

    def update(self, force: bool = False) -> Dict[str, float]:
        size = len(self.buffer)
        if size < (8 if force else self.batch_size):
            return {
                "role_actor_loss": 0.0,
                "role_critic_loss": 0.0,
                "role_entropy": 0.0,
                "role_approx_kl": 0.0,
            }

        role_obs = torch.as_tensor(
            np.asarray(self.buffer.role_obs), dtype=torch.float32, device=self.device
        )
        global_states = torch.as_tensor(
            np.asarray(self.buffer.global_states), dtype=torch.float32, device=self.device
        )
        next_global_states = torch.as_tensor(
            np.asarray(self.buffer.next_global_states), dtype=torch.float32, device=self.device
        )
        joint_masks = torch.as_tensor(
            np.asarray(self.buffer.joint_masks), dtype=torch.bool, device=self.device
        )
        actions = torch.as_tensor(
            self.buffer.joint_actions, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            self.buffer.option_rewards, dtype=torch.float32, device=self.device
        )
        durations = torch.as_tensor(
            self.buffer.durations, dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            self.buffer.terminated, dtype=torch.bool, device=self.device
        )
        truncated = torch.as_tensor(
            self.buffer.truncated, dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            values = self.critic(global_states).squeeze(-1)
            next_values = self.critic(next_global_states).squeeze(-1)
            advantages, returns = self.compute_smdp_gae(
                rewards,
                durations,
                terminated,
                truncated,
                values,
                next_values,
            )
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        updates = 0
        mini_batch = min(self.batch_size, size)
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(size, device=self.device)
            for start in range(0, size, mini_batch):
                mb = indices[start : start + mini_batch]
                distribution = self._joint_distribution(role_obs[mb], joint_masks[mb])
                new_log_probs = distribution.log_prob(actions[mb])
                entropy = distribution.entropy().mean()
                log_ratio = new_log_probs - old_log_probs[mb]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantages[mb]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                value_pred = self.critic(global_states[mb]).squeeze(-1)
                critic_loss = self.value_coef * F.mse_loss(value_pred, returns[mb])

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                actor_loss_sum += float(actor_loss.item())
                critic_loss_sum += float(critic_loss.item())
                entropy_sum += float(entropy.item())
                kl_sum += float(approx_kl.item())
                updates += 1

        self.buffer.clear()
        denom = max(updates, 1)
        return {
            "role_actor_loss": actor_loss_sum / denom,
            "role_critic_loss": critic_loss_sum / denom,
            "role_entropy": entropy_sum / denom,
            "role_approx_kl": kl_sum / denom,
        }


class Stage4StepMixScheduler:
    RAMP_UPDATES = 10

    def __init__(self, start_update: int):
        self.rehearsal_steps = 0
        self.continuation_steps = 0
        self.post_target_steps = 0
        self._start_ratio = 0.0
        self._target_ratio = 0.0
        self._ramp_start_update = int(start_update)

    def set_target_ratio(self, target_ratio: float, update_step: int):
        target_ratio = float(np.clip(target_ratio, 0.0, 1.0))
        if np.isclose(target_ratio, self._target_ratio):
            return
        current_ratio = self.current_target_ratio(update_step)
        self._start_ratio = current_ratio
        self._target_ratio = target_ratio
        self._ramp_start_update = int(update_step)

    def current_target_ratio(self, update_step: int) -> float:
        progress = np.clip(
            (int(update_step) - self._ramp_start_update) / float(self.RAMP_UPDATES),
            0.0,
            1.0,
        )
        return float(self._start_ratio + progress * (self._target_ratio - self._start_ratio))

    def choose_mode(self, update_step: int) -> str:
        ratio = self.current_target_ratio(update_step)
        if ratio <= 0.0:
            return "rehearsal"
        if ratio >= 1.0:
            return "continuation"
        rehearsal_progress = self.rehearsal_steps / (1.0 - ratio)
        continuation_progress = self.continuation_steps / ratio
        return "continuation" if continuation_progress < rehearsal_progress else "rehearsal"

    def record(self, mode: str, episode_steps: int, first_target_step: int = -1):
        episode_steps = max(0, int(episode_steps))
        if mode == "continuation":
            self.continuation_steps += episode_steps
            if int(first_target_step) > 0:
                self.post_target_steps += max(0, episode_steps - int(first_target_step))
        else:
            self.rehearsal_steps += episode_steps

    def stats(self, update_step: int) -> Dict[str, float]:
        total = self.rehearsal_steps + self.continuation_steps
        return {
            "target_continuation_ratio": self.current_target_ratio(update_step),
            "realized_continuation_ratio": self.continuation_steps / max(total, 1),
            "realized_post_target_ratio": self.post_target_steps / max(total, 1),
            "rehearsal_steps": int(self.rehearsal_steps),
            "continuation_steps": int(self.continuation_steps),
            "post_target_steps": int(self.post_target_steps),
        }


class CurriculumManager:
    STAGE2_SUBSTAGES = ("2A", "2B", "2C")
    STAGE2_MIN_EPISODES = (150, 150, 200)
    STAGE3_SUBSTAGES = ("3A", "3B", "3C", "3D")
    STAGE3_TARGET_MIN_EPS = (150, 200, 250, 300)
    STAGE3_CONTINUATION_RATIOS = (0.10, 0.15, 0.20, 0.30)
    STAGE3_EXTRA_STEPS = (50, 75, 100, 150)
    STAGE4_CONTINUATION_RATIOS = (0.40, 0.70, 1.00)
    STAGE4_EXTRA_STEPS = (150, 300, None)
    STAGE4_GOALS = (0.60, 0.60, 0.60)
    VALIDATION_WINDOW = 3
    VALIDATION_REQUIRED_PASSES = 2
    STAGE4_MIN_REMAINING_EPISODES = 0

    def __init__(self, final_init_area_percent: float = 5.0, stage3_final_target: float = 0.60):
        self.current_stage = 1
        self.stage_episodes = {1: 0, 2: 0, 3: 0, 4: 0}
        self.stage_success_rates = {
            stage: deque(maxlen=50) for stage in self.stage_episodes
        }
        self.stage_coverages = {
            stage: deque(maxlen=50) for stage in self.stage_episodes
        }
        self.stage_zero_timeout_rates = {
            stage: deque(maxlen=50) for stage in self.stage_episodes
        }
        self._fixed_init_area_percent = float(final_init_area_percent)
        self._stage2_idx = 0
        self._s3_target_idx = 0
        self.STAGE3_TARGET_LADDER = [0.20, 0.35, 0.50, float(stage3_final_target)]
        self.substage_episodes = {
            "1": 0,
            **{name: 0 for name in self.STAGE2_SUBSTAGES},
            **{name: 0 for name in self.STAGE3_SUBSTAGES},
            "4A": 0,
            "4B": 0,
            "4C": 0,
        }
        self._validation_history = deque(maxlen=self.VALIDATION_WINDOW)
        self._terminal_focus_active = False
        self._stage4_level_idx = 0
        self._stage4_progress_history = deque(maxlen=self.VALIDATION_WINDOW)
        self._stage4_guard_history = deque(maxlen=self.VALIDATION_WINDOW)
        self._stage4_stop_requested = False
        self._stage4_recovery_requested = False
        self._stage4_baseline = None
        self._stage3_ready_for_stage4 = False

    @property
    def current_init_percentile(self):
        return self._fixed_init_area_percent

    @property
    def current_substage(self) -> str:
        if self.current_stage == 1:
            return "1"
        if self.current_stage == 2:
            return self.STAGE2_SUBSTAGES[self._stage2_idx]
        if self.current_stage == 3:
            return self.STAGE3_SUBSTAGES[self._s3_target_idx]
        return self.stage4_level

    @property
    def current_stage3_target(self) -> float:
        return self.STAGE3_TARGET_LADDER[self._s3_target_idx]

    @property
    def stage3_near_prob(self) -> float:
        return 0.0

    @property
    def stage3_continuation_ratio(self) -> float:
        return self.STAGE3_CONTINUATION_RATIOS[self._s3_target_idx]

    @property
    def stage3_extra_steps(self):
        return self.STAGE3_EXTRA_STEPS[self._s3_target_idx]

    @property
    def stage4_level(self) -> str:
        return ("4A", "4B", "4C")[self._stage4_level_idx]

    @property
    def stage4_continuation_ratio(self) -> float:
        return self.STAGE4_CONTINUATION_RATIOS[self._stage4_level_idx]

    @property
    def stage4_extra_steps(self):
        return self.STAGE4_EXTRA_STEPS[self._stage4_level_idx]

    @property
    def stage4_goal(self) -> float:
        return self.STAGE4_GOALS[self._stage4_level_idx]

    def update(self, success: bool, coverage: float, zero_coverage_timeout: bool = False) -> int:
        stage = self.current_stage
        self.stage_episodes[stage] += 1
        self.substage_episodes[self.current_substage] += 1
        self.stage_success_rates[stage].append(1.0 if success else 0.0)
        self.stage_coverages[stage].append(float(coverage))
        self.stage_zero_timeout_rates[stage].append(1.0 if zero_coverage_timeout else 0.0)
        return self.current_stage

    def _minimum_episodes_reached(self) -> bool:
        if self.current_stage == 1:
            required = 200
        elif self.current_stage == 2:
            required = self.STAGE2_MIN_EPISODES[self._stage2_idx]
        elif self.current_stage == 3:
            required = self.STAGE3_TARGET_MIN_EPS[self._s3_target_idx]
        else:
            required = (200, 250, 300)[self._stage4_level_idx]
        return self.substage_episodes[self.current_substage] >= required

    def _validation_passed(self, summary: Dict) -> bool:
        found = float(summary.get("boundary_found_rate", 0.0))
        zero_timeout = float(
            summary.get(
                "zero_discovery_timeout_rate",
                summary.get("zero_coverage_timeout_rate", 0.0),
            )
        )
        success = float(summary.get("success_rate", 0.0))
        if self.current_stage == 1:
            median_contact = summary.get("median_first_boundary_step")
            return bool(
                found >= 0.90
                and zero_timeout <= 0.10
                and float(summary.get("stable_tracking_success_rate", 0.0)) >= 0.80
                and median_contact is not None
                and float(median_contact) <= 80.0
                and float(summary.get("median_unique_boundary_cells", 0.0)) >= 12.0
            )
        if self.current_stage == 2:
            median_contact = summary.get("median_first_boundary_step")
            return bool(
                found >= 0.85
                and success >= 0.75
                and zero_timeout <= 0.10
                and median_contact is not None
                and float(median_contact) <= 150.0
            )
        if self.current_stage == 3:
            required_success = 0.70 if self._s3_target_idx < 2 else 0.65
            return bool(
                found >= 0.85
                and success >= required_success
                and zero_timeout <= 0.10
            )
        return False

    def update_validation(self, summary: Dict) -> bool:
        if self.current_stage == 4 or not self._minimum_episodes_reached():
            return False
        self._validation_history.append(self._validation_passed(summary))
        if (
            len(self._validation_history) < self.VALIDATION_WINDOW
            or sum(self._validation_history) < self.VALIDATION_REQUIRED_PASSES
        ):
            return False

        old_substage = self.current_substage
        self._validation_history.clear()
        if self.current_stage == 1:
            self.current_stage = 2
        elif self.current_stage == 2:
            if self._stage2_idx < len(self.STAGE2_SUBSTAGES) - 1:
                self._stage2_idx += 1
            else:
                self.current_stage = 3
        elif self._s3_target_idx < len(self.STAGE3_TARGET_LADDER) - 1:
            self._s3_target_idx += 1
        else:
            self._stage3_ready_for_stage4 = True
        next_label = (
            "Stage4准入就绪" if self._stage3_ready_for_stage4 else self.current_substage
        )
        print(
            f"\n  [验证课程] {old_substage} -> {next_label} | "
            f"validation_passes={self.VALIDATION_REQUIRED_PASSES}/{self.VALIDATION_WINDOW}"
        )
        return True

    def can_enter_stage4(self, validation_summary: Dict) -> bool:
        return bool(
            self.current_stage == 3
            and self._stage3_ready_for_stage4
            and self._s3_target_idx == len(self.STAGE3_TARGET_LADDER) - 1
            and float(validation_summary.get("success_rate", 0.0)) >= 0.65
            and float(validation_summary.get("zero_coverage_timeout_rate", 1.0)) <= 0.10
        )

    def enter_stage4(self, full_horizon_baseline: Dict):
        self.current_stage = 4
        self._stage4_level_idx = 0
        self._stage4_progress_history.clear()
        self._stage4_guard_history.clear()
        self._stage4_stop_requested = False
        self._stage4_recovery_requested = False
        self._stage4_baseline = dict(full_horizon_baseline)

    def update_stage4_validation(self, summary: Dict) -> bool:
        if self.current_stage != 4 or self._stage4_baseline is None:
            return False
        baseline = self._stage4_baseline
        guard_passed = bool(
            float(summary.get("boundary_found_rate", 0.0))
            >= max(0.85, float(baseline.get("boundary_found_rate", 0.0)) - 0.05)
            and float(summary.get("target_reach_rate", 0.0))
            >= max(0.0, float(baseline.get("target_reach_rate", 0.0)) - 0.05)
        )
        self._stage4_guard_history.append(guard_passed)
        if not guard_passed:
            self._stage4_progress_history.append(False)
            if (
                len(self._stage4_guard_history) == self.VALIDATION_WINDOW
                and not any(self._stage4_guard_history)
            ):
                self._stage4_level_idx = max(0, self._stage4_level_idx - 1)
                self._stage4_recovery_requested = True
                self._stage4_guard_history.clear()
                self._stage4_progress_history.clear()
            return False
        if self._stage4_level_idx >= len(self.STAGE4_CONTINUATION_RATIOS) - 1:
            return False
        hold = float(summary.get("mean_hold_ratio_by_threshold", {}).get("0.60", 0.0))
        baseline_hold = float(
            baseline.get("mean_hold_ratio_by_threshold", {}).get("0.60", 0.0)
        )
        if self._stage4_level_idx == 0:
            improvement_passed = (
                float(summary.get("mean_tail100_coverage", 0.0))
                >= float(baseline.get("mean_tail100_coverage", 0.0))
                and hold >= baseline_hold
            )
        elif self._stage4_level_idx == 1:
            improvement_passed = (
                float(summary.get("mean_tail100_coverage", 0.0))
                >= float(baseline.get("mean_tail100_coverage", 0.0)) + 0.02
                and float(summary.get("mean_current_coverage_auc", 0.0))
                >= float(baseline.get("mean_current_coverage_auc", 0.0))
            )
        else:
            improvement_passed = (
                float(summary.get("mean_tail100_coverage", 0.0))
                >= float(baseline.get("mean_tail100_coverage", 0.0)) + 0.03
                and float(summary.get("mean_current_coverage_auc", 0.0))
                >= float(baseline.get("mean_current_coverage_auc", 0.0)) + 0.01
                and hold >= baseline_hold
            )

        self._stage4_progress_history.append(bool(improvement_passed))
        if (
            len(self._stage4_progress_history) < self.VALIDATION_WINDOW
            or sum(self._stage4_progress_history) < self.VALIDATION_REQUIRED_PASSES
        ):
            return False
        self._stage4_level_idx += 1
        self._stage4_progress_history.clear()
        self._stage4_guard_history.clear()
        return True

    def update_stage3_validation(self, summary: Dict) -> bool:
        return self.update_validation(summary) if self.current_stage == 3 else False

    def state_dict(self) -> Dict:
        return {
            "current_stage": self.current_stage,
            "current_substage": self.current_substage,
            "stage_episodes": dict(self.stage_episodes),
            "substage_episodes": dict(self.substage_episodes),
            "stage2_index": self._stage2_idx,
            "stage3_target_index": self._s3_target_idx,
            "stage4_level_index": self._stage4_level_idx,
            "stage3_ready_for_stage4": self._stage3_ready_for_stage4,
            "stage4_baseline": copy.deepcopy(self._stage4_baseline),
        }

    def get_stage_info(self) -> Dict:
        stage = self.current_stage
        success_rate = float(np.mean(self.stage_success_rates[stage])) if self.stage_success_rates[stage] else 0.0
        return {
            "stage": stage,
            "substage": self.current_substage,
            "episodes": self.stage_episodes[stage],
            "success_rate": success_rate,
            "total_episodes": sum(self.stage_episodes.values()),
            "init_area_percent": self.current_init_percentile,
            "stage3_target": self.current_stage3_target if stage == 3 else None,
            "stage3_near_prob": self.stage3_near_prob if stage == 3 else None,
            "stage4_level": self.stage4_level if stage == 4 else None,
            "stage4_continuation_ratio": self.stage4_continuation_ratio if stage == 4 else None,
            "stage4_extra_steps": self.stage4_extra_steps if stage == 4 else None,
            "stage4_goal": self.stage4_goal if stage == 4 else None,
        }

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
        target_kl: float = 0.0065,
        actor_lr_min: float = 1e-4,
        actor_lr_max: float = 2.5e-4,
        kl_ema_beta: float = 0.8,
        kl_lr_low_ratio: float = 0.82,
        kl_lr_high_ratio: float = 1.20,
        kl_lr_emergency_ratio: float = 2.00,
        kl_lr_up_factor: float = 1.03,
        kl_lr_down_factor: float = 0.90,
        kl_lr_emergency_factor: float = 0.70,
        kl_lr_low_patience: int = 3,
        kl_early_stop_ratio: float = 1.50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 4096,
        hierarchical_roles_enabled: bool = False,
        role_obs_dim: int = 0,
        num_roles: int = 3,
        role_actor_lr: float = 2e-4,
        role_critic_lr: float = 5e-4,
        role_batch_size: int = 128,
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
        self.kl_lr_low_ratio = max(0.0, float(kl_lr_low_ratio))
        self.kl_lr_high_ratio = max(self.kl_lr_low_ratio, float(kl_lr_high_ratio))
        self.kl_lr_emergency_ratio = max(self.kl_lr_high_ratio, float(kl_lr_emergency_ratio))
        self.kl_lr_up_factor = max(1.0, float(kl_lr_up_factor))
        self.kl_lr_down_factor = float(np.clip(kl_lr_down_factor, 1e-6, 1.0))
        self.kl_lr_emergency_factor = float(
            np.clip(kl_lr_emergency_factor, 1e-6, self.kl_lr_down_factor)
        )
        self.kl_lr_low_patience = max(1, int(kl_lr_low_patience))
        self.kl_early_stop_ratio = max(1.0, float(kl_early_stop_ratio))
        self.kl_ema = None
        self._consecutive_low_kl = 0
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
        self.hierarchical_roles_enabled = bool(hierarchical_roles_enabled)
        self.role_agent = (
            RolePPOAgent(
                role_obs_dim=role_obs_dim,
                global_state_dim=global_state_dim,
                num_roles=num_roles,
                device=self.device,
                actor_lr=role_actor_lr,
                critic_lr=role_critic_lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                max_grad_norm=max_grad_norm,
                ppo_epochs=ppo_epochs,
                batch_size=role_batch_size,
            )
            if self.hierarchical_roles_enabled
            else None
        )
        self.training_step = 0

        print(
            f"CTDE-PPO 基线已初始化 | 设备={self.device} | "
            f"本地观测={local_obs_dim} | 全局状态={global_state_dim} | "
            f"lr_adapt_mode={self.lr_adapt_mode}"
        )

    def select_roles_if_required(
        self,
        obs: Dict,
        deterministic: bool = False,
        track_option: bool = True,
    ):
        if self.role_agent is None or not bool(obs.get("role_decision_required", 0)):
            return None
        roles, log_prob, joint_action = self.role_agent.select_joint_roles(
            obs["role_obs"],
            obs["joint_role_mask"],
            deterministic=deterministic,
        )
        if track_option:
            self.role_agent.begin_option(
                obs["role_obs"],
                obs["global_state"],
                obs["joint_role_mask"],
                joint_action,
                log_prob,
            )
        return roles

    def imitate_roles(self, obs: Dict, roles: List[int]) -> float:
        if self.role_agent is None:
            return 0.0
        return self.role_agent.imitate_joint_roles(
            obs["role_obs"], obs["joint_role_mask"], roles
        )

    def accumulate_role_reward(self, rewards: List[float]) -> None:
        if self.role_agent is not None:
            self.role_agent.accumulate_reward(float(np.mean(rewards)))

    def finish_role_option(self, next_obs: Dict, info: Dict) -> None:
        if self.role_agent is None:
            return
        self.role_agent.finish_option(
            next_obs["global_state"],
            terminated=bool(info.get("terminated", False)),
            truncated=bool(info.get("truncated", False)),
        )

    def update_roles(self, force: bool = False) -> Dict[str, float]:
        if self.role_agent is None:
            return {
                "role_actor_loss": 0.0,
                "role_critic_loss": 0.0,
                "role_entropy": 0.0,
                "role_approx_kl": 0.0,
            }
        return self.role_agent.update(force=force)

    def _set_actor_lr(self, lr: float):
        lr = float(np.clip(lr, self.actor_lr_min, self.actor_lr_max))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr

    def _set_critic_lr(self, lr: float):
        for group in self.critic_optimizer.param_groups:
            group["lr"] = float(lr)

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
        if mean_kl > self.kl_lr_emergency_ratio * self.target_kl:
            self._consecutive_low_kl = 0
            new_lr = current_lr * self.kl_lr_emergency_factor
            action = "emergency_down"
        elif max(mean_kl, kl_ema) > self.kl_lr_high_ratio * self.target_kl:
            self._consecutive_low_kl = 0
            new_lr = current_lr * self.kl_lr_down_factor
            action = "down"
        elif kl_ema < self.kl_lr_low_ratio * self.target_kl:
            self._consecutive_low_kl += 1
            if self._consecutive_low_kl >= self.kl_lr_low_patience:
                self._consecutive_low_kl = 0
                new_lr = current_lr * self.kl_lr_up_factor
                action = "up"
            else:
                new_lr = current_lr
                action = "low_wait"
        else:
            self._consecutive_low_kl = 0
            new_lr = current_lr
            action = "keep"

        self._set_actor_lr(new_lr)
        if not np.isclose(self.actor_optimizer.param_groups[0]["lr"], current_lr):
            return action
        return "keep" if action in {"up", "down", "emergency_down"} else action

    def select_actions(
        self,
        local_obs: List[np.ndarray],
        action_masks: List[np.ndarray] = None,
    ) -> Tuple[List[int], List[float]]:
        local_obs_tensor = torch.FloatTensor(np.array(local_obs)).to(self.device)
        action_masks_tensor = None
        if action_masks is not None:
            action_masks_tensor = torch.as_tensor(
                np.asarray(action_masks), dtype=torch.bool, device=self.device
            )
        with torch.no_grad():
            action_probs = self.actor.get_action_probs(
                local_obs_tensor, action_masks_tensor
            )
            actions = action_probs.sample()
            log_probs = action_probs.log_prob(actions)
        return actions.cpu().numpy().tolist(), log_probs.cpu().numpy().tolist()

    def select_actions_deterministic(
        self,
        local_obs: List[np.ndarray],
        action_masks: List[np.ndarray] = None,
    ) -> List[int]:
        local_obs_tensor = torch.FloatTensor(np.array(local_obs)).to(self.device)
        action_masks_tensor = None
        if action_masks is not None:
            action_masks_tensor = torch.as_tensor(
                np.asarray(action_masks), dtype=torch.bool, device=self.device
            )
        with torch.no_grad():
            logits = self.actor(local_obs_tensor)
            if action_masks_tensor is not None:
                logits = logits.masked_fill(
                    ~action_masks_tensor, torch.finfo(logits.dtype).min
                )
            actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy().tolist()

    def store_transition(
        self,
        local_obs,
        global_state,
        actions,
        log_probs,
        rewards,
        done,
        next_global_state=None,
        truncated=False,
        mode=0,
        action_masks=None,
    ):
        if action_masks is None:
            action_masks = [
                np.ones(self.actor.action_head.out_features, dtype=np.int8)
                for _ in actions
            ]
        self.buffer.store(
            local_obs,
            global_state,
            actions,
            log_probs,
            rewards,
            done,
            next_global_state=next_global_state,
            truncated=truncated,
            mode=mode,
            action_masks=action_masks,
        )

    def compute_gae(
        self,
        rewards_list: List[List[float]],
        terminated: List[bool],
        truncated: List[bool],
        global_states: np.ndarray,
        next_global_states: np.ndarray,
    ):
        global_states_tensor = torch.FloatTensor(global_states).to(self.device)
        next_global_states_tensor = torch.FloatTensor(next_global_states).to(self.device)
        with torch.no_grad():
            values = self.critic(global_states_tensor).squeeze(-1)
            next_values = self.critic(next_global_states_tensor).squeeze(-1)

        team_rewards = [float(np.mean(rewards)) for rewards in rewards_list]
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(team_rewards))):
            bootstrap_mask = 1.0 - float(terminated[t])
            trace_mask = 1.0 - float(terminated[t] or truncated[t])
            delta = team_rewards[t] + self.gamma * next_values[t] * bootstrap_mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * trace_mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.stack(advantages), torch.stack(returns)

    def update(self, force: bool = False) -> Dict[str, float]:
        required_batch = self.min_update_batch_size if force else self.batch_size
        buffer_size = len(self.buffer)
        if buffer_size < required_batch:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        (
            local_obs_list,
            global_states,
            next_global_states,
            actions_list,
            action_masks_list,
            old_log_probs_list,
            rewards_list,
            terminated,
            truncated,
            modes,
        ) = self.buffer.get()
        global_states_np = np.array(global_states)
        next_global_states_np = np.array(next_global_states)
        advantages, returns = self.compute_gae(
            rewards_list,
            terminated,
            truncated,
            global_states_np,
            next_global_states_np,
        )
        mode_array = np.asarray(modes, dtype=np.int64)
        normalized_advantages = advantages.clone()
        for mode in np.unique(mode_array):
            mode_indices_np = np.flatnonzero(mode_array == mode)
            mode_indices = torch.as_tensor(mode_indices_np, dtype=torch.long, device=self.device)
            mode_advantages = advantages[mode_indices]
            if mode_advantages.numel() > 1:
                normalized_advantages[mode_indices] = (
                    mode_advantages - mode_advantages.mean()
                ) / (mode_advantages.std() + 1e-8)
            else:
                normalized_advantages[mode_indices] = 0.0
        advantages = normalized_advantages

        global_states_tensor = torch.FloatTensor(global_states_np).to(self.device)
        local_obs_tensor = torch.FloatTensor(np.array(local_obs_list)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions_list)).to(self.device)
        action_masks_tensor = torch.as_tensor(
            np.asarray(action_masks_list), dtype=torch.bool, device=self.device
        )
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs_list)).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        update_steps = 0
        ppo_epochs_completed = 0
        kl_early_stop = False

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
                mb_action_masks = action_masks_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]

                obs_dim = mb_local_obs.shape[-1]
                flat_obs = mb_local_obs.view(-1, obs_dim)
                flat_actions = mb_actions.view(-1)
                flat_action_masks = mb_action_masks.view(
                    -1, mb_action_masks.shape[-1]
                )
                flat_old_log_probs = mb_old_log_probs.view(-1)
                flat_advantages = mb_advantages.unsqueeze(1).expand(-1, self.num_agents).reshape(-1)

                action_probs = self.actor.get_action_probs(flat_obs, flat_action_masks)
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

            ppo_epochs_completed += 1
            running_mean_kl = total_approx_kl / max(update_steps, 1)
            if (
                self.lr_adapt_mode == "kl"
                and running_mean_kl > self.kl_early_stop_ratio * self.target_kl
            ):
                kl_early_stop = True
                break

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
            "target_kl": self.target_kl,
            "consecutive_low_kl": self._consecutive_low_kl,
            "kl_early_stop": kl_early_stop,
            "ppo_epochs_completed": ppo_epochs_completed,
            "clip_fraction": total_clip_fraction / denom,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "entropy_coef": self.entropy_coef,
        }

    def save(self, path: str, training_state: Dict = None):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
            "kl_ema": self.kl_ema,
            "lr_controller_state": {
                "kl_ema": self.kl_ema,
                "consecutive_low_kl": self._consecutive_low_kl,
                "target_kl": self.target_kl,
            },
            "hierarchical_roles_enabled": self.hierarchical_roles_enabled,
        }
        if self.role_agent is not None:
            checkpoint.update(
                {
                    "role_actor_state_dict": self.role_agent.actor.state_dict(),
                    "role_critic_state_dict": self.role_agent.critic.state_dict(),
                    "role_actor_optimizer_state_dict": self.role_agent.actor_optimizer.state_dict(),
                    "role_critic_optimizer_state_dict": self.role_agent.critic_optimizer.state_dict(),
                }
            )
        if training_state is not None:
            checkpoint["training_state"] = copy.deepcopy(training_state)
        torch.save(checkpoint, path)

    def load(self, path: str, restore_training_state: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if self.role_agent is not None:
            if "role_actor_state_dict" not in checkpoint:
                raise ValueError(
                    "checkpoint does not contain the hierarchical role policy"
                )
            self.role_agent.actor.load_state_dict(checkpoint["role_actor_state_dict"])
            self.role_agent.critic.load_state_dict(checkpoint["role_critic_state_dict"])
        if not restore_training_state:
            self.kl_ema = None
            self._consecutive_low_kl = 0
            return checkpoint

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if self.role_agent is not None:
            self.role_agent.actor_optimizer.load_state_dict(
                checkpoint["role_actor_optimizer_state_dict"]
            )
            self.role_agent.critic_optimizer.load_state_dict(
                checkpoint["role_critic_optimizer_state_dict"]
            )
        self._set_actor_lr(self.actor_optimizer.param_groups[0]["lr"])
        self.training_step = int(checkpoint.get("training_step", 0))
        controller_state = checkpoint.get("lr_controller_state", {})
        self.kl_ema = controller_state.get("kl_ema", checkpoint.get("kl_ema"))
        self._consecutive_low_kl = int(controller_state.get("consecutive_low_kl", 0))
        if "target_kl" in controller_state:
            self.target_kl = max(1e-8, float(controller_state["target_kl"]))
        return checkpoint


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
        "communication_enabled": bool(config["communication_enabled"]),
        "communication_radius_factor": float(config["communication_radius_factor"]),
        "communication_radius": float(
            env_info.get(
                "communication_radius",
                config["communication_radius_factor"] * config["vision_radius"],
            )
        ),
        "action_mask_enabled": bool(config["action_mask_enabled"]),
        "hierarchical_roles_enabled": bool(config["hierarchical_roles_enabled"]),
        "peer_state_ttl": int(config["peer_state_ttl"]),
        "track_report_ttl": int(config["track_report_ttl"]),
        "reacquire_report_ttl": int(config["reacquire_report_ttl"]),
        "role_decision_interval": int(config["role_decision_interval"]),
        "role_min_dwell_steps": int(config["role_min_dwell_steps"]),
        "boundary_match_radius": int(config["boundary_match_radius"]),
        "boundary_freshness_tau": float(config["boundary_freshness_tau"]),
        "mask_thermal_below_signal": bool(config["mask_thermal_below_signal"]),
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
    evaluation_mode: str = "target_stop",
) -> Dict:
    eval_config = copy.deepcopy(base_config)
    eval_config["eval_split"] = str(split).lower()
    eval_config["eval_scene_keys"] = None
    eval_config["eval_episodes_per_scene"] = int(episodes_per_scene)
    eval_config["eval_stages"] = [int(stage) for stage in stages]
    eval_config["evaluation_mode"] = str(evaluation_mode).lower()
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


def _full_horizon_episode_metrics(
    coverage_history: List[float],
    info: Dict,
    target: float,
    thresholds: List[float],
    coverage_curve: List[Dict],
) -> Dict:
    coverage = np.asarray(coverage_history, dtype=np.float64)
    if coverage.size == 0:
        coverage = np.asarray([0.0], dtype=np.float64)
    tail = coverage[-min(100, coverage.size):]
    first_target_step = int(info.get("first_target_step", -1))
    if first_target_step > 0:
        post_target = coverage[first_target_step - 1:]
        post_target_tail = post_target[-min(100, post_target.size):]
        target_hold_ratio = float(np.mean(post_target >= float(target)))
        hold_ratio_by_threshold = {
            f"{float(threshold):.2f}": float(np.mean(post_target >= float(threshold)))
            for threshold in thresholds
        }
        post_target_peak_gain = float(max(0.0, np.max(post_target) - float(target)))
        post_target_tail_gain = float(np.mean(post_target_tail) - float(target))
    else:
        target_hold_ratio = 0.0
        hold_ratio_by_threshold = {
            f"{float(threshold):.2f}": 0.0 for threshold in thresholds
        }
        post_target_peak_gain = 0.0
        post_target_tail_gain = 0.0

    threshold_steps = {}
    for threshold in thresholds:
        hits = np.flatnonzero(coverage >= float(threshold))
        threshold_steps[f"{float(threshold):.2f}"] = int(hits[0] + 1) if hits.size else -1

    refresh_points = [point for point in coverage_curve if point.get("boundary_refreshed")]
    final_refresh = refresh_points[-1] if refresh_points else {}

    return {
        "current_coverage_auc": float(np.mean(coverage)),
        "tail100_mean_coverage": float(np.mean(tail)),
        "final_current_coverage": float(coverage[-1]),
        "max_current_coverage": float(np.max(coverage)),
        "historical_boundary_union_coverage": float(
            info.get("historical_boundary_union_coverage", 0.0)
        ),
        "target_reached": bool(info.get("target_reached", False)),
        "first_target_step": first_target_step,
        "target_hold_ratio": target_hold_ratio,
        "hold_ratio_by_threshold": hold_ratio_by_threshold,
        "post_target_peak_gain": post_target_peak_gain,
        "post_target_tail_gain": post_target_tail_gain,
        "last_boundary_refresh_step": int(final_refresh.get("step", -1)),
        "coverage_before_final_refresh": float(
            final_refresh.get("coverage_before_refresh", coverage[-1])
        ),
        "coverage_after_final_refresh": float(
            final_refresh.get("coverage_after_refresh", coverage[-1])
        ),
        "threshold_steps": threshold_steps,
        "coverage_curve": coverage_curve,
    }


def _summarize_full_horizon_records(records: List[Dict], target: float) -> Dict:
    def mean_field(key: str) -> float:
        return float(np.mean([float(record.get(key, 0.0)) for record in records])) if records else 0.0

    reached_steps = [
        int(record["first_target_step"])
        for record in records
        if int(record.get("first_target_step", -1)) > 0
    ]
    threshold_keys = sorted(
        {key for record in records for key in record.get("threshold_steps", {}).keys()}
    )
    threshold_reach_rate = {}
    mean_time_to_threshold = {}
    mean_hold_ratio_by_threshold = {}
    for key in threshold_keys:
        steps = [int(record["threshold_steps"].get(key, -1)) for record in records]
        reached = [step for step in steps if step > 0]
        threshold_reach_rate[key] = float(len(reached) / len(records)) if records else 0.0
        mean_time_to_threshold[key] = float(np.mean(reached)) if reached else None
        mean_hold_ratio_by_threshold[key] = float(
            np.mean(
                [record.get("hold_ratio_by_threshold", {}).get(key, 0.0) for record in records]
            )
        ) if records else 0.0

    return {
        "evaluation_mode": "full_horizon",
        "episodes": len(records),
        "target": float(target),
        "target_reach_rate": mean_field("target_reached"),
        "mean_first_target_step": float(np.mean(reached_steps)) if reached_steps else None,
        "mean_current_coverage_auc": mean_field("current_coverage_auc"),
        "mean_tail100_coverage": mean_field("tail100_mean_coverage"),
        "mean_final_current_coverage": mean_field("final_current_coverage"),
        "mean_max_current_coverage": mean_field("max_current_coverage"),
        "mean_historical_boundary_union_coverage": mean_field(
            "historical_boundary_union_coverage"
        ),
        "mean_target_hold_ratio": mean_field("target_hold_ratio"),
        "mean_post_target_peak_gain": mean_field("post_target_peak_gain"),
        "mean_post_target_tail_gain": mean_field("post_target_tail_gain"),
        "mean_pre_boundary_revisit_ratio": mean_field("pre_boundary_revisit_ratio"),
        "mean_team_overlap_ratio": mean_field("team_overlap_ratio"),
        "mean_communication_available_rate": mean_field("communication_available_rate"),
        "mean_invalid_action_count": mean_field("invalid_action_count"),
        "mean_refresh_recovery_rate": float(
            np.sum([record.get("refresh_recovery_successes", 0) for record in records])
            / max(np.sum([record.get("major_refresh_count", 0) for record in records]), 1)
        ),
        "mean_refresh_recovery_time": float(
            np.mean(
                [
                    record["mean_refresh_recovery_time"]
                    for record in records
                    if record.get("mean_refresh_recovery_time") is not None
                ]
            )
        ) if any(record.get("mean_refresh_recovery_time") is not None for record in records) else None,
        "boundary_found_rate": float(
            np.mean([int(record.get("first_boundary_step", -1)) > 0 for record in records])
        ) if records else 0.0,
        "zero_discovery_timeout_rate": mean_field("zero_discovery_timeout"),
        "horizon_completion_rate": float(
            np.mean([record.get("done_reason") == "horizon_reached" for record in records])
        ) if records else 0.0,
        "battery_depleted_rate": float(
            np.mean([record.get("done_reason") == "battery_depleted" for record in records])
        ) if records else 0.0,
        "mean_length": mean_field("length"),
        "threshold_reach_rate": threshold_reach_rate,
        "mean_time_to_threshold": mean_time_to_threshold,
        "mean_hold_ratio_by_threshold": mean_hold_ratio_by_threshold,
        "records": records,
    }


def _post_target_gate_passed(summary: Dict, goal: float) -> bool:
    if float(summary.get("boundary_found_rate", 0.0)) < 0.95:
        return False
    hold = summary.get("mean_hold_ratio_by_threshold", {})
    reach = summary.get("threshold_reach_rate", {})
    if goal < 0.625:
        return (
            float(summary.get("mean_tail100_coverage", 0.0)) >= 0.62
            and float(hold.get("0.60", 0.0)) >= 0.55
        )
    if goal < 0.675:
        return (
            float(summary.get("mean_tail100_coverage", 0.0)) >= 0.67
            and float(hold.get("0.65", 0.0)) >= 0.50
            and float(reach.get("0.70", 0.0)) >= 0.45
        )
    return False


def _advance_post_target_goal(
    goal_index: int,
    consecutive_passes: int,
    summary: Dict,
    ladder: List[float],
    patience: int,
) -> Tuple[int, int, bool]:
    if goal_index >= len(ladder) - 1:
        return goal_index, 0, False
    consecutive_passes = (
        consecutive_passes + 1
        if _post_target_gate_passed(summary, ladder[goal_index])
        else 0
    )
    if consecutive_passes >= patience:
        return goal_index + 1, 0, True
    return goal_index, consecutive_passes, False


def _post_target_optimizer_settings(update_index: int, config: Dict) -> Dict[str, float]:
    warmup_updates = int(config["post_target_warmup_updates"])
    if update_index < warmup_updates:
        progress = 1.0 if warmup_updates <= 1 else update_index / (warmup_updates - 1)
        actor_lr = config["post_target_actor_lr"] * (0.5 + 0.5 * progress)
        critic_lr = config["post_target_critic_lr"] * (0.5 + 0.5 * progress)
        return {
            "actor_lr": float(actor_lr),
            "critic_lr": float(critic_lr),
            "clip_epsilon": float(config["post_target_initial_clip_epsilon"]),
            "max_grad_norm": float(config["post_target_initial_max_grad_norm"]),
            "override_actor_lr": True,
        }
    return {
        "actor_lr": float(config["post_target_actor_lr"]),
        "critic_lr": float(config["post_target_critic_lr"]),
        "clip_epsilon": float(config["clip_epsilon"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "override_actor_lr": config["lr_adapt_mode"] == "fixed",
    }


def _after_train_eval_checkpoints(config: Dict, best_model_paths: Dict) -> List[Tuple[str, str]]:
    best_stage4_path = best_model_paths.get("best_stage4")
    if best_stage4_path and os.path.exists(best_stage4_path):
        return [("best_stage4", best_stage4_path)]
    best_val_path = best_model_paths.get("best_val")
    if (
        config["evaluate_best_val_after_train"]
        and best_val_path
        and os.path.exists(best_val_path)
    ):
        return [("best_val", best_val_path)]
    return []


def _persistent_env_kwargs(config: Dict) -> Dict:
    return {
        "hierarchical_roles_enabled": config["hierarchical_roles_enabled"],
        "peer_state_ttl": config["peer_state_ttl"],
        "track_report_ttl": config["track_report_ttl"],
        "reacquire_report_ttl": config["reacquire_report_ttl"],
        "role_decision_interval": config["role_decision_interval"],
        "role_min_dwell_steps": config["role_min_dwell_steps"],
        "boundary_match_radius": config["boundary_match_radius"],
        "boundary_freshness_tau": config["boundary_freshness_tau"],
        "fresh_coverage_gain_weight": config["fresh_coverage_gain_weight"],
        "assigned_boundary_gain_weight": config["assigned_boundary_gain_weight"],
        "role_switch_penalty": config["role_switch_penalty"],
        "mask_thermal_below_signal": config["mask_thermal_below_signal"],
        "post_target_step_cost_fraction": config["post_target_step_cost_fraction"],
    }


def _role_agent_kwargs(config: Dict, env: FireSearchBaselineEnvironment) -> Dict:
    env_info = env.get_env_info()
    return {
        "hierarchical_roles_enabled": config["hierarchical_roles_enabled"],
        "role_obs_dim": env_info["role_obs_dim"],
        "num_roles": env_info["num_roles"],
        "role_actor_lr": config["role_actor_lr"],
        "role_critic_lr": config["role_critic_lr"],
        "role_batch_size": config["role_batch_size"],
    }


def _make_eval_agent(config: Dict, env: FireSearchBaselineEnvironment) -> CTDE_PPO_Agent:
    return CTDE_PPO_Agent(
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
        kl_lr_low_ratio=config["kl_lr_low_ratio"],
        kl_lr_high_ratio=config["kl_lr_high_ratio"],
        kl_lr_emergency_ratio=config["kl_lr_emergency_ratio"],
        kl_lr_up_factor=config["kl_lr_up_factor"],
        kl_lr_down_factor=config["kl_lr_down_factor"],
        kl_lr_emergency_factor=config["kl_lr_emergency_factor"],
        kl_lr_low_patience=config["kl_lr_low_patience"],
        kl_early_stop_ratio=config["kl_early_stop_ratio"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_coef"],
        value_coef=config["value_coef"],
        max_grad_norm=config["max_grad_norm"],
        ppo_epochs=config["ppo_epochs"],
        batch_size=config["batch_size"],
        **_role_agent_kwargs(config, env),
    )


def _full_horizon_checkpoint_candidates(
    model_dir: str,
    final_episode: int,
    best_val_path: str = None,
    stage3_source_path: str = None,
    best_stage4_path: str = None,
) -> List[Tuple[str, str]]:
    stage4_candidates = []
    if stage3_source_path and os.path.exists(stage3_source_path):
        stage4_candidates.append(("stage3_source", stage3_source_path))
    if best_stage4_path and os.path.exists(best_stage4_path):
        stage4_candidates.append(("best_stage4", best_stage4_path))
    if stage4_candidates:
        return stage4_candidates

    first_terminal_episode = max(1, int(final_episode) - 300)
    candidates = []
    for filename in os.listdir(model_dir):
        if not filename.startswith("ppo_ep") or "_stage3.pth" not in filename:
            continue
        episode_text = filename[len("ppo_ep"):].split("_stage", 1)[0]
        if not episode_text.isdigit():
            continue
        episode = int(episode_text)
        if episode >= first_terminal_episode:
            candidates.append((f"ep{episode}", os.path.join(model_dir, filename)))
    candidates.sort(key=lambda item: int(item[0][2:]))
    if not candidates and best_val_path and os.path.exists(best_val_path):
        candidates.append(("best_val", best_val_path))
    return candidates


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
        f"actor_lr范围=[{config['actor_lr_min']:.2e}, {config['actor_lr_max']:.2e}]"
    )
    print(
        f"KL迟滞区间=[{config['kl_lr_low_ratio']:.2f}, {config['kl_lr_high_ratio']:.2f}] | "
        f"紧急阈值={config['kl_lr_emergency_ratio']:.2f} | "
        f"epoch熔断阈值={config['kl_early_stop_ratio']:.2f}"
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
    continuation_scheduler = Stage4StepMixScheduler(start_update=0)
    env = FireSearchBaselineEnvironment(
        data_dir=config["data_dir"],
        num_drones=config["num_drones"],
        vision_radius=config["vision_radius"],
        max_steps=config["max_steps"],
        use_metadata_uav_params=config["use_metadata_uav_params"],
        observation_profile=config["observation_profile"],
        reward_profile=config["reward_profile"],
        communication_enabled=config["communication_enabled"],
        communication_radius_factor=config["communication_radius_factor"],
        action_mask_enabled=config["action_mask_enabled"],
        novelty_reward_weight=config["novelty_reward_weight"],
        novelty_step_penalty=config["novelty_step_penalty"],
        novelty_revisit_penalty=config["novelty_revisit_penalty"],
        invalid_action_penalty=config["invalid_action_penalty"],
        team_overlap_penalty=config["team_overlap_penalty"],
        curriculum_stage=1,
        mode=config["train_split"],
        scene_keys=config["train_scene_keys"],
        init_percentile=config["init_percentile"],
        init_area_percent=curriculum.current_init_percentile,
        stage2_target=config["stage2_success_target"],
        stage3_target=curriculum.current_stage3_target,
        stage3_near_prob=curriculum.stage3_near_prob,
        **_persistent_env_kwargs(config),
    )
    env.set_curriculum_substage(curriculum.current_substage)
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
        kl_lr_low_ratio=config["kl_lr_low_ratio"],
        kl_lr_high_ratio=config["kl_lr_high_ratio"],
        kl_lr_emergency_ratio=config["kl_lr_emergency_ratio"],
        kl_lr_up_factor=config["kl_lr_up_factor"],
        kl_lr_down_factor=config["kl_lr_down_factor"],
        kl_lr_emergency_factor=config["kl_lr_emergency_factor"],
        kl_lr_low_patience=config["kl_lr_low_patience"],
        kl_early_stop_ratio=config["kl_early_stop_ratio"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_coef"],
        value_coef=config["value_coef"],
        max_grad_norm=config["max_grad_norm"],
        ppo_epochs=config["ppo_epochs"],
        batch_size=config["batch_size"],
        **_role_agent_kwargs(config, env),
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
        "communication_available_rate": [],
        "shared_new_cells": [],
        "pre_boundary_revisit_ratio": [],
        "team_overlap_ratio": [],
        "invalid_action_count": [],
        "reward_breakdown": [],
        "stage": [],
        "substage": [],
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
        "target_kl": [],
        "consecutive_low_kl": [],
        "kl_early_stop": [],
        "ppo_epochs_completed": [],
        "clip_fraction": [],
        "actor_lr": [],
        "critic_lr": [],
        "init_area_percent": [],
        "stage3_target": [],
        "stage3_near_prob": [],
        "terminal_focus": [],
        "stage4_level": [],
        "stage4_mode": [],
        "stage4_extra_steps": [],
        "stage4_target_continuation_ratio": [],
        "stage4_realized_continuation_ratio": [],
        "stage4_realized_post_target_ratio": [],
    }

    validation_log = {
        "episodes": [],
        "stage": [],
        "substage": [],
        "train_task_score": [],
        "val_mean_task_score": [],
        "val_mean_coverage": [],
        "val_success_rate": [],
        "val_mean_length": [],
        "val_timeout_rate": [],
        "val_zero_coverage_timeout_rate": [],
        "val_boundary_found_rate": [],
        "val_pre_boundary_revisit_ratio": [],
        "val_team_overlap_ratio": [],
        "val_invalid_action_count": [],
        "generalization_gap": [],
        "is_best_val": [],
        "stage4_level": [],
        "stage4_boundary_found_rate": [],
        "stage4_target_reach_rate": [],
        "stage4_auc": [],
        "stage4_tail100": [],
        "stage4_hold60": [],
    }

    start_time = time.time()
    total_steps = 0
    best_task_score = -float("inf")
    best_val_model_score = -float("inf")
    best_stage4_score = -float("inf")
    best_model_paths = {
        "best": None,
        "best_train": None,
        "best_val": None,
        "stage3_source": None,
        "best_stage4": None,
        "final": None,
    }
    update_info = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "kl_ema": 0.0,
        "kl_lr_action": "fixed" if config["lr_adapt_mode"] == "fixed" else "keep",
        "target_kl": config["target_kl"],
        "consecutive_low_kl": 0,
        "kl_early_stop": False,
        "ppo_epochs_completed": 0,
        "clip_fraction": 0.0,
        "actor_lr": config["actor_lr"],
        "critic_lr": config["critic_lr"],
    }

    def checkpoint_training_state(episode_number: int) -> Dict:
        return {
            "episode": int(episode_number),
            "total_environment_steps": int(total_steps),
            "curriculum": curriculum.state_dict(),
        }

    for episode in range(1, config["total_episodes"] + 1):
        remaining = config["total_episodes"] - episode
        desired_env_stage = 3 if curriculum.current_stage == 4 else curriculum.current_stage
        if env.curriculum_stage != desired_env_stage:
            env.set_curriculum_stage(desired_env_stage)
        env.set_curriculum_substage(curriculum.current_substage)
        env.stage_targets[3] = curriculum.current_stage3_target
        env.stage3_near_prob = 0.0
        stage4_mode = "baseline"
        buffer_mode = 0
        if curriculum.current_stage in {3, 4}:
            target_ratio = (
                curriculum.stage3_continuation_ratio
                if curriculum.current_stage == 3
                else curriculum.stage4_continuation_ratio
            )
            extra_steps = (
                curriculum.stage3_extra_steps
                if curriculum.current_stage == 3
                else curriculum.stage4_extra_steps
            )
            continuation_scheduler.set_target_ratio(target_ratio, agent.training_step)
            stage4_mode = continuation_scheduler.choose_mode(agent.training_step)
            if stage4_mode == "continuation":
                env.termination_mode = "post_target_train"
                env.set_post_target_extra_steps(extra_steps)
                env.set_post_target_goal(
                    curriculum.current_stage3_target
                    if curriculum.current_stage == 3
                    else curriculum.stage4_goal
                )
                buffer_mode = 1
            else:
                env.termination_mode = "target_stop"
                env.set_post_target_extra_steps(None)
        else:
            env.termination_mode = "target_stop"
            env.set_post_target_extra_steps(None)
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        role_imitation_losses = []

        while not done:
            train_role_policy = curriculum.current_stage >= 3
            if bool(obs.get("role_decision_required", 0)):
                if train_role_policy:
                    roles = agent.select_roles_if_required(obs, deterministic=False)
                else:
                    roles = env.heuristic_joint_role_assignment()
                    role_imitation_losses.append(agent.imitate_roles(obs, roles))
                obs = env.apply_joint_role_assignment(roles)
            local_obs = obs["local_obs"]
            global_state = obs["global_state"]
            action_masks = obs["action_masks"]
            actions, log_probs = agent.select_actions(local_obs, action_masks)
            next_obs, rewards, done, info = env.step(actions)
            if train_role_policy:
                agent.accumulate_role_reward(rewards)
                if done or bool(next_obs.get("role_decision_required", 0)):
                    agent.finish_role_option(next_obs, info)
            agent.store_transition(
                local_obs,
                global_state,
                actions,
                log_probs,
                rewards,
                done,
                next_global_state=next_obs["global_state"],
                truncated=bool(info.get("truncated", False)),
                mode=buffer_mode,
                action_masks=action_masks,
            )

            episode_reward += float(sum(rewards))
            episode_length += 1
            total_steps += 1
            obs = next_obs

        if len(agent.buffer) >= config["batch_size"]:
            update_info = agent.update()
        role_update_info = agent.update_roles() if curriculum.current_stage >= 3 else {}

        if curriculum.current_stage in {3, 4}:
            continuation_scheduler.record(
                stage4_mode,
                episode_length,
                first_target_step=int(info.get("first_target_step", -1)),
            )
        success = (
            bool(info.get("target_reached", False))
            if curriculum.current_stage in {3, 4}
            else info["done_reason"] == "mission_complete"
        )
        timeout = bool(
            info["done_reason"] in {"max_steps_reached", "horizon_reached"}
            and not info.get("target_reached", False)
        )
        zero_timeout = bool(info.get("zero_coverage_timeout", False))
        objective_coverage = float(
            info.get("objective_coverage", info["boundary_coverage"])
        )
        task_score = _task_score(objective_coverage, success, episode_length, config["max_steps"])

        rolling_rewards.append(episode_reward)
        rolling_lengths.append(episode_length)
        rolling_coverages.append(objective_coverage)
        rolling_success.append(1.0 if success else 0.0)
        rolling_task_scores.append(task_score)
        rolling_timeouts.append(1.0 if timeout else 0.0)
        rolling_zero_timeouts.append(1.0 if zero_timeout else 0.0)

        training_log["episodes"].append(episode)
        training_log["rewards"].append(episode_reward)
        training_log["task_scores"].append(task_score)
        training_log["lengths"].append(episode_length)
        training_log["coverages"].append(objective_coverage)
        training_log["success"].append(1 if success else 0)
        training_log["done_reasons"].append(info.get("done_reason", "other"))
        training_log["timeout"].append(1 if timeout else 0)
        training_log["zero_coverage_timeout"].append(1 if zero_timeout else 0)
        _append_episode_diagnostics(training_log, info)
        training_log["stage"].append(curriculum.current_stage)
        training_log["substage"].append(curriculum.current_substage)
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
        training_log["target_kl"].append(update_info.get("target_kl", config["target_kl"]))
        training_log["consecutive_low_kl"].append(update_info.get("consecutive_low_kl", 0))
        training_log["kl_early_stop"].append(bool(update_info.get("kl_early_stop", False)))
        training_log["ppo_epochs_completed"].append(update_info.get("ppo_epochs_completed", 0))
        training_log["clip_fraction"].append(update_info.get("clip_fraction", 0.0))
        training_log["actor_lr"].append(update_info.get("actor_lr", config["actor_lr"]))
        training_log["critic_lr"].append(update_info.get("critic_lr", config["critic_lr"]))
        training_log.setdefault("role_actor_loss", []).append(
            role_update_info.get("role_actor_loss", 0.0)
        )
        training_log.setdefault("role_critic_loss", []).append(
            role_update_info.get("role_critic_loss", 0.0)
        )
        training_log.setdefault("role_entropy", []).append(
            role_update_info.get("role_entropy", 0.0)
        )
        training_log.setdefault("role_approx_kl", []).append(
            role_update_info.get("role_approx_kl", 0.0)
        )
        training_log.setdefault("role_imitation_loss", []).append(
            float(np.mean(role_imitation_losses)) if role_imitation_losses else 0.0
        )
        training_log["init_area_percent"].append(env.init_area_percent)
        training_log["stage3_target"].append(env.stage_targets[3])
        training_log["stage3_near_prob"].append(env.stage3_near_prob)
        training_log["terminal_focus"].append(
            1 if curriculum._terminal_focus_active else 0
        )
        if curriculum.current_stage in {3, 4}:
            stage4_stats = continuation_scheduler.stats(agent.training_step)
            training_log["stage4_level"].append(
                curriculum.stage4_level if curriculum.current_stage == 4 else None
            )
            training_log["stage4_mode"].append(stage4_mode)
            training_log["stage4_extra_steps"].append(
                curriculum.stage4_extra_steps
                if curriculum.current_stage == 4
                else curriculum.stage3_extra_steps
            )
            training_log["stage4_target_continuation_ratio"].append(
                stage4_stats["target_continuation_ratio"]
            )
            training_log["stage4_realized_continuation_ratio"].append(
                stage4_stats["realized_continuation_ratio"]
            )
            training_log["stage4_realized_post_target_ratio"].append(
                stage4_stats["realized_post_target_ratio"]
            )
        else:
            training_log["stage4_level"].append(None)
            training_log["stage4_mode"].append("baseline")
            training_log["stage4_extra_steps"].append(None)
            training_log["stage4_target_continuation_ratio"].append(0.0)
            training_log["stage4_realized_continuation_ratio"].append(0.0)
            training_log["stage4_realized_post_target_ratio"].append(0.0)

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
        env_stage = 3 if new_stage == 4 else new_stage
        if env_stage != env.curriculum_stage or difficulty_changed:
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
        if env_stage != env.curriculum_stage:
            env.set_curriculum_stage(env_stage)
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
            validated_stage = curriculum.current_stage
            validated_substage = curriculum.current_substage
            validated_stage4_level = (
                curriculum.stage4_level if curriculum.current_stage == 4 else None
            )
            stage4_validation = curriculum.current_stage == 4
            eval_stage = 3 if stage4_validation else curriculum.current_stage
            val_config = make_eval_config(
                config,
                config["validation_split"],
                max(10, config["validation_episodes_per_scene"])
                if stage4_validation
                else config["validation_episodes_per_scene"],
                [eval_stage],
                evaluation_mode="full_horizon" if stage4_validation else "target_stop",
            )
            val_config["curriculum_substage"] = curriculum.current_substage
            val_config["stage3_success_target"] = curriculum.current_stage3_target
            print("\n" + "=" * 70)
            print(
                f"验证评估 | 回合={episode} | "
                f"阶段={curriculum.current_stage} | "
                f"每场景回合={max(10, config['validation_episodes_per_scene']) if stage4_validation else config['validation_episodes_per_scene']}"
            )
            print("=" * 70)
            validation_results = evaluate_preserving_rng(agent, val_config)
            val_summary = validation_results[int(eval_stage)]
            train_mean_task_score = (
                float(np.mean(rolling_task_scores)) if rolling_task_scores else task_score
            )
            if stage4_validation:
                val_score = float(val_summary["mean_current_coverage_auc"])
                hold60 = float(
                    val_summary.get("mean_hold_ratio_by_threshold", {}).get("0.60", 0.0)
                )
                stage4_score = float(
                    0.30 * val_summary["mean_current_coverage_auc"]
                    + 0.25 * val_summary["mean_tail100_coverage"]
                    + 0.20 * val_summary.get("mean_target_hold_ratio", 0.0)
                    + 0.15 * val_summary.get("mean_refresh_recovery_rate", 0.0)
                    + 0.10 * val_summary.get("target_reach_rate", 0.0)
                )
                baseline = curriculum._stage4_baseline
                guard_passed = (
                    float(val_summary.get("boundary_found_rate", 0.0))
                    >= max(0.85, float(baseline.get("boundary_found_rate", 0.0)) - 0.05)
                    and float(val_summary.get("target_reach_rate", 0.0))
                    >= max(0.0, float(baseline.get("target_reach_rate", 0.0)) - 0.05)
                )
                is_best_val = bool(
                    config["save_best_by_validation"]
                    and guard_passed
                    and stage4_score > best_stage4_score
                )
                if is_best_val:
                    best_stage4_score = stage4_score
                    best_stage4_path = os.path.join(model_dir, "ppo_best_stage4.pth")
                    agent.save(best_stage4_path, checkpoint_training_state(episode))
                    best_model_paths["best_stage4"] = best_stage4_path
                    print(f"  -> 最佳阶段4验证分数: {best_stage4_score * 100:.1f}%")

                old_level = curriculum.stage4_level
                level_advanced = curriculum.update_stage4_validation(val_summary)
                if level_advanced:
                    if len(agent.buffer) >= agent.min_update_batch_size:
                        pending_update_info = agent.update(force=True)
                        if pending_update_info.get("actor_loss", 0.0) != 0.0:
                            update_info = pending_update_info
                    else:
                        agent.buffer.clear()
                    continuation_scheduler = Stage4StepMixScheduler(agent.training_step)
                    continuation_scheduler.set_target_ratio(
                        curriculum.stage4_continuation_ratio,
                        agent.training_step,
                    )
                    print(
                        f"  [stage4 curriculum] {old_level} -> {curriculum.stage4_level} | "
                        f"continuation_ratio={curriculum.stage4_continuation_ratio:.2f} | "
                        f"extra_steps={curriculum.stage4_extra_steps}"
                    )
                if curriculum._stage4_recovery_requested:
                    safe_path = best_model_paths.get("best_stage4") or best_model_paths.get(
                        "stage3_source"
                    )
                    if safe_path:
                        agent.load(safe_path, restore_training_state=False)
                    current_lr = agent.actor_optimizer.param_groups[0]["lr"]
                    reduced_lr = (
                        max(agent.actor_lr_min, current_lr * 0.70)
                        if agent.lr_adapt_mode == "kl"
                        else current_lr
                    )
                    agent._set_actor_lr(reduced_lr)
                    agent.kl_ema = None
                    agent._consecutive_low_kl = 0
                    continuation_scheduler = Stage4StepMixScheduler(agent.training_step)
                    continuation_scheduler.set_target_ratio(
                        curriculum.stage4_continuation_ratio,
                        agent.training_step,
                    )
                    curriculum._stage4_recovery_requested = False
                    lr_note = (
                        f"actor_lr降至{reduced_lr:.2e}"
                        if agent.lr_adapt_mode == "kl"
                        else f"固定actor_lr保持{reduced_lr:.2e}"
                    )
                    print(
                        "  [stage4 recovery] 连续三次低于保护线，已回滚安全模型、"
                        f"降低难度并{lr_note}，训练继续"
                    )

                val_mean_coverage = float(val_summary["mean_final_current_coverage"])
                val_success_rate = float(val_summary["target_reach_rate"])
                val_mean_length = float(val_summary["mean_length"])
                val_timeout_rate = 1.0 - float(val_summary["horizon_completion_rate"])
                val_zero_timeout_rate = 0.0
                val_boundary_found_rate = float(val_summary["boundary_found_rate"])
                val_pre_boundary_revisit_ratio = float(
                    val_summary.get("mean_pre_boundary_revisit_ratio", 0.0)
                )
                val_team_overlap_ratio = float(
                    val_summary.get("mean_team_overlap_ratio", 0.0)
                )
                val_invalid_action_count = float(
                    val_summary.get("mean_invalid_action_count", 0.0)
                )
                stage4_boundary_found = float(val_summary["boundary_found_rate"])
                stage4_target_reach = float(val_summary["target_reach_rate"])
                stage4_auc = float(val_summary["mean_current_coverage_auc"])
                stage4_tail100 = float(val_summary["mean_tail100_coverage"])
                stage4_hold60 = hold60
            else:
                val_score = float(val_summary["mean_task_score"])
                val_model_score = _validation_model_score(val_summary)
                is_best_val = (
                    bool(config["save_best_by_validation"])
                    and curriculum.current_stage == 3
                    and curriculum._s3_target_idx
                    == len(curriculum.STAGE3_TARGET_LADDER) - 1
                    and val_model_score > best_val_model_score
                )
                if is_best_val:
                    best_val_model_score = val_model_score
                    best_val_path = os.path.join(model_dir, "ppo_best_val.pth")
                    agent.save(best_val_path, checkpoint_training_state(episode))
                    best_model_paths["best_val"] = best_val_path
                    print(
                        f"  -> 最佳验证模型分数: {best_val_model_score * 100:.1f}% "
                        f"(任务得分={val_score * 100:.1f}%)"
                    )
                val_mean_coverage = float(val_summary["mean_coverage"])
                val_success_rate = float(val_summary["success_rate"])
                val_mean_length = float(val_summary["mean_length"])
                val_timeout_rate = float(val_summary["timeout_rate"])
                val_zero_timeout_rate = float(val_summary["zero_coverage_timeout_rate"])
                val_boundary_found_rate = float(val_summary["boundary_found_rate"])
                val_pre_boundary_revisit_ratio = float(
                    val_summary["mean_pre_boundary_revisit_ratio"]
                )
                val_team_overlap_ratio = float(val_summary["mean_team_overlap_ratio"])
                val_invalid_action_count = float(val_summary["mean_invalid_action_count"])
                stage4_boundary_found = 0.0
                stage4_target_reach = 0.0
                stage4_auc = 0.0
                stage4_tail100 = 0.0
                stage4_hold60 = 0.0

                substage_checkpoint_state = checkpoint_training_state(episode)
                if curriculum.update_validation(val_summary):
                    substage_key = (
                        "stage1" if validated_substage == "1" else validated_substage.lower()
                    )
                    substage_path = os.path.join(
                        model_dir, f"ppo_{substage_key}_best.pth"
                    )
                    agent.save(substage_path, substage_checkpoint_state)
                    best_model_paths[f"{substage_key}_best"] = substage_path
                    if len(agent.buffer) >= agent.min_update_batch_size:
                        pending_update_info = agent.update(force=True)
                        if pending_update_info.get("actor_loss", 0.0) != 0.0:
                            update_info = pending_update_info
                    else:
                        agent.buffer.clear()
                    env.stage3_near_prob = curriculum.stage3_near_prob
                    env.stage_targets[3] = curriculum.current_stage3_target
                    env.set_curriculum_substage(curriculum.current_substage)
                    continuation_scheduler = Stage4StepMixScheduler(agent.training_step)
                    if curriculum.current_stage == 3:
                        continuation_scheduler.set_target_ratio(
                            curriculum.stage3_continuation_ratio,
                            agent.training_step,
                        )

            validation_log["episodes"].append(episode)
            validation_log["stage"].append(validated_stage)
            validation_log["substage"].append(validated_substage)
            validation_log["train_task_score"].append(train_mean_task_score)
            validation_log["val_mean_task_score"].append(val_score)
            validation_log["val_mean_coverage"].append(val_mean_coverage)
            validation_log["val_success_rate"].append(val_success_rate)
            validation_log["val_mean_length"].append(val_mean_length)
            validation_log["val_timeout_rate"].append(val_timeout_rate)
            validation_log["val_zero_coverage_timeout_rate"].append(val_zero_timeout_rate)
            validation_log["val_boundary_found_rate"].append(val_boundary_found_rate)
            validation_log["val_pre_boundary_revisit_ratio"].append(
                val_pre_boundary_revisit_ratio
            )
            validation_log["val_team_overlap_ratio"].append(val_team_overlap_ratio)
            validation_log["val_invalid_action_count"].append(val_invalid_action_count)
            validation_log["generalization_gap"].append(train_mean_task_score - val_score)
            validation_log["is_best_val"].append(1 if is_best_val else 0)
            validation_log["stage4_level"].append(
                validated_stage4_level
            )
            validation_log["stage4_boundary_found_rate"].append(stage4_boundary_found)
            validation_log["stage4_target_reach_rate"].append(stage4_target_reach)
            validation_log["stage4_auc"].append(stage4_auc)
            validation_log["stage4_tail100"].append(stage4_tail100)
            validation_log["stage4_hold60"].append(stage4_hold60)

            if (
                not stage4_validation
                and remaining >= CurriculumManager.STAGE4_MIN_REMAINING_EPISODES
                and curriculum.can_enter_stage4(val_summary)
            ):
                if len(agent.buffer) >= agent.min_update_batch_size:
                    pending_update_info = agent.update(force=True)
                    if pending_update_info.get("actor_loss", 0.0) != 0.0:
                        update_info = pending_update_info
                else:
                    agent.buffer.clear()
                baseline_config = make_eval_config(
                    config,
                    config["validation_split"],
                    max(10, config["validation_episodes_per_scene"]),
                    [3],
                    evaluation_mode="full_horizon",
                )
                baseline_results = evaluate_preserving_rng(agent, baseline_config)
                full_horizon_baseline = baseline_results[3]
                source_path = os.path.join(model_dir, "ppo_stage3_source.pth")
                agent.save(source_path, checkpoint_training_state(episode))
                best_model_paths["stage3_source"] = source_path
                curriculum.enter_stage4(full_horizon_baseline)
                continuation_scheduler = Stage4StepMixScheduler(agent.training_step)
                continuation_scheduler.set_target_ratio(
                    curriculum.stage4_continuation_ratio,
                    agent.training_step,
                )
                env.set_curriculum_stage(3)
                env.stage3_near_prob = 0.0
                print(
                    f"  [stage4 entry] 进入4A | source={source_path} | "
                    f"continuation_ratio={curriculum.stage4_continuation_ratio:.2f} | "
                    f"extra_steps={curriculum.stage4_extra_steps}"
                )

        if episode % config["save_interval"] == 0:
            checkpoint_path = os.path.join(model_dir, f"ppo_ep{episode}_stage{curriculum.current_stage}.pth")
            agent.save(checkpoint_path, checkpoint_training_state(episode))
            mean_task_score = float(np.mean(rolling_task_scores)) if rolling_task_scores else task_score
            if mean_task_score > best_task_score:
                best_task_score = mean_task_score
                best_path = os.path.join(model_dir, "ppo_best_train.pth")
                agent.save(best_path, checkpoint_training_state(episode))
                legacy_best_path = os.path.join(model_dir, "ppo_best.pth")
                agent.save(legacy_best_path, checkpoint_training_state(episode))
                best_model_paths["best"] = best_path
                best_model_paths["best_train"] = best_path
                print(f"  -> 最佳训练滚动任务得分: {best_task_score * 100:.1f}%")

        max_train_updates = config.get("max_train_updates")
        if max_train_updates is not None and agent.training_step >= max_train_updates:
            print(f"达到 PPO 更新预算: {agent.training_step}/{max_train_updates}")
            break
        max_environment_steps = config.get("max_environment_steps")
        if max_environment_steps is not None and total_steps >= max_environment_steps:
            print(f"达到环境步预算: {total_steps}/{max_environment_steps}")
            break

    update_limit_reached = (
        config["max_train_updates"] is not None
        and agent.training_step >= config["max_train_updates"]
    )
    if len(agent.buffer) >= agent.min_update_batch_size and not update_limit_reached:
        update_info = agent.update(force=True)
    if curriculum.current_stage == 4:
        agent.update_roles(force=True)

    final_episode = int(training_log["episodes"][-1]) if training_log["episodes"] else 0
    final_model_path = os.path.join(model_dir, "ppo_final.pth")
    agent.save(final_model_path, checkpoint_training_state(final_episode))
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
    full_horizon_summary_path = None
    eval_summary = {
        "best_val": {
            "available": bool(best_model_paths.get("best_val")),
            "model_path": best_model_paths.get("best_val"),
            "splits": {},
        },
        "best_stage4": {
            "available": bool(best_model_paths.get("best_stage4")),
            "model_path": best_model_paths.get("best_stage4"),
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
            best_val_agent = _make_eval_agent(config, env)
            best_val_agent.load(checkpoint_path, restore_training_state=False)

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

    if config["full_horizon_eval_after_train"]:
        print("\n" + "=" * 70)
        print("训练后完整时域覆盖评估")
        print(
            f"validation选模每场景回合={config['full_horizon_validation_episodes_per_scene']} | "
            f"最终报告每场景回合={config['full_horizon_final_episodes_per_scene']}"
        )
        print("=" * 70)

        candidates = _full_horizon_checkpoint_candidates(
            model_dir,
            final_episode,
            best_model_paths.get("best_val"),
            best_model_paths.get("stage3_source"),
            best_model_paths.get("best_stage4"),
        )
        selection_stage = 3 if 3 in config["eval_stages"] else int(config["eval_stages"][-1])
        full_horizon_summary = {
            "evaluation_mode": "full_horizon",
            "selection_split": config["validation_split"],
            "selection_metric": "mean_current_coverage_auc",
            "selection_stage": selection_stage,
            "candidates": [],
            "selected": None,
            "splits": {},
        }

        if not candidates:
            print("未找到终末阶段 checkpoint，跳过完整时域评估")
        else:
            coverage_agent = _make_eval_agent(config, env)
            best_candidate = None
            best_candidate_score = (-float("inf"), -float("inf"))
            for candidate_name, candidate_path in candidates:
                coverage_agent.load(candidate_path, restore_training_state=False)
                selection_config = make_eval_config(
                    config,
                    config["validation_split"],
                    config["full_horizon_validation_episodes_per_scene"],
                    [selection_stage],
                    evaluation_mode="full_horizon",
                )
                print(f"\n完整时域 checkpoint 选模 | {candidate_name}")
                candidate_results = evaluate(coverage_agent, selection_config)
                candidate_summary = _stage_summary(candidate_results, selection_stage)
                candidate_eval_path = os.path.join(
                    log_dir,
                    f"full_horizon_candidate_{candidate_name}_validation_eval.json",
                )
                _write_json(candidate_eval_path, candidate_results)
                candidate_entry = {
                    "name": candidate_name,
                    "model_path": candidate_path,
                    "eval_path": candidate_eval_path,
                    "summary": candidate_summary,
                }
                full_horizon_summary["candidates"].append(candidate_entry)
                candidate_score = (
                    float(candidate_summary["mean_current_coverage_auc"]),
                    float(candidate_summary["mean_tail100_coverage"]),
                )
                if candidate_score > best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate = candidate_entry

            selected_model_path = os.path.join(model_dir, "ppo_best_full_coverage.pth")
            shutil.copy2(best_candidate["model_path"], selected_model_path)
            best_model_paths["best_full_coverage"] = selected_model_path
            full_horizon_summary["selected"] = {
                **best_candidate,
                "model_path": selected_model_path,
            }
            eval_summary["best_full_coverage"] = {
                "available": True,
                "model_path": selected_model_path,
                "selection": full_horizon_summary["selected"],
                "splits": {},
            }

            coverage_agent.load(selected_model_path, restore_training_state=False)
            for split in config["full_horizon_eval_splits"]:
                split_config = make_eval_config(
                    config,
                    split,
                    config["full_horizon_final_episodes_per_scene"],
                    config["eval_stages"],
                    evaluation_mode="full_horizon",
                )
                print(f"\nbest_full_coverage 完整时域评估 | 数据划分={split}")
                split_results = evaluate(coverage_agent, split_config)
                split_path = os.path.join(
                    log_dir,
                    f"best_full_coverage_full_horizon_{split}_eval.json",
                )
                _write_json(split_path, split_results)
                split_entry = {
                    "path": split_path,
                    "stages": {
                        str(stage): _stage_summary(split_results, int(stage))
                        for stage in config["eval_stages"]
                    },
                }
                full_horizon_summary["splits"][split] = split_entry
                eval_summary["best_full_coverage"]["splits"][split] = split_entry

        full_horizon_summary_path = os.path.join(log_dir, "full_horizon_summary.json")
        _write_json(full_horizon_summary_path, full_horizon_summary)
        print(f"full_horizon_summary={full_horizon_summary_path}")

    if config["eval_after_train"] or config["full_horizon_eval_after_train"]:
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
    if full_horizon_summary_path:
        print(f"完整时域评估摘要={full_horizon_summary_path}")
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


def train_post_target(config: Dict = None, checkpoint_path: str = None):
    config = normalize_training_config(config)
    checkpoint_path = checkpoint_path or config.get("post_target_resume_checkpoint")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("post-target training requires an existing checkpoint")

    config["post_target_train"] = True
    config["post_target_resume_checkpoint"] = os.path.abspath(checkpoint_path)
    output_dir = _make_output_dir(config)
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    setup_console_tee(os.path.join(output_dir, CONSOLE_LOG_NAME), mode="w")
    set_seed(config["seed"])

    dataset_index = _resolve_dataset_scene_keys(config)
    ladder = list(config["post_target_goal_ladder"])
    thresholds = sorted(
        set(config["full_horizon_thresholds"] + ladder + [0.60, 0.65, 0.70, 0.80])
    )
    config["full_horizon_thresholds"] = thresholds

    env = FireSearchBaselineEnvironment(
        data_dir=config["data_dir"],
        num_drones=config["num_drones"],
        vision_radius=config["vision_radius"],
        max_steps=config["max_steps"],
        use_metadata_uav_params=config["use_metadata_uav_params"],
        observation_profile=config["observation_profile"],
        reward_profile=config["reward_profile"],
        communication_enabled=config["communication_enabled"],
        communication_radius_factor=config["communication_radius_factor"],
        action_mask_enabled=config["action_mask_enabled"],
        novelty_reward_weight=config["novelty_reward_weight"],
        novelty_step_penalty=config["novelty_step_penalty"],
        novelty_revisit_penalty=config["novelty_revisit_penalty"],
        invalid_action_penalty=config["invalid_action_penalty"],
        team_overlap_penalty=config["team_overlap_penalty"],
        curriculum_stage=3,
        mode=config["train_split"],
        scene_keys=config["train_scene_keys"],
        init_percentile=config["init_percentile"],
        init_area_percent=config["init_area_percent"],
        stage2_target=config["stage2_success_target"],
        stage3_target=config["stage3_success_target"],
        stage3_near_prob=0.0,
        termination_mode="post_target_train",
        post_target_goal=ladder[0],
        post_target_step_penalty=config["post_target_step_penalty"],
        post_target_hold_weight=config["post_target_hold_weight"],
        post_target_tail_weight=config["post_target_tail_weight"],
        post_target_milestone_70=config["post_target_milestone_70"],
        post_target_milestone_80=config["post_target_milestone_80"],
        **_persistent_env_kwargs(config),
    )
    config["experiment_metadata"] = _build_experiment_metadata(
        config, dataset_index, env.get_env_info()
    )
    _write_json(os.path.join(output_dir, "config.json"), config)

    agent = CTDE_PPO_Agent(
        local_obs_dim=env.local_obs_dim,
        global_state_dim=env.global_state_dim,
        action_dim=env.num_actions,
        num_agents=env.num_drones,
        actor_lr=config["post_target_actor_lr"],
        critic_lr=config["post_target_critic_lr"],
        lr_adapt_mode=config["lr_adapt_mode"],
        target_kl=config["target_kl"],
        actor_lr_min=min(config["actor_lr_min"], config["post_target_actor_lr"] * 0.5),
        actor_lr_max=config["actor_lr_max"],
        kl_ema_beta=config["kl_ema_beta"],
        kl_lr_low_ratio=config["kl_lr_low_ratio"],
        kl_lr_high_ratio=config["kl_lr_high_ratio"],
        kl_lr_emergency_ratio=config["kl_lr_emergency_ratio"],
        kl_lr_up_factor=config["kl_lr_up_factor"],
        kl_lr_down_factor=config["kl_lr_down_factor"],
        kl_lr_emergency_factor=config["kl_lr_emergency_factor"],
        kl_lr_low_patience=config["kl_lr_low_patience"],
        kl_early_stop_ratio=config["kl_early_stop_ratio"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_coef"],
        value_coef=config["value_coef"],
        max_grad_norm=config["max_grad_norm"],
        ppo_epochs=config["ppo_epochs"],
        batch_size=config["batch_size"],
        **_role_agent_kwargs(config, env),
    )
    checkpoint = agent.load(checkpoint_path, restore_training_state=False)
    saved_state = checkpoint.get("training_state", {})
    resuming_post_target = saved_state.get("phase") == "post_target"
    if resuming_post_target:
        checkpoint = agent.load(checkpoint_path, restore_training_state=True)
        saved_state = checkpoint.get("training_state", {})

    start_episode = int(saved_state.get("post_target_episode", 0)) + 1
    goal_index = min(int(saved_state.get("goal_index", 0)), len(ladder) - 1)
    consecutive_passes = int(saved_state.get("consecutive_validation_passes", 0))
    env.set_post_target_goal(ladder[goal_index])
    agent.buffer.clear()

    def training_state(episode: int) -> Dict:
        return {
            "phase": "post_target",
            "post_target_episode": int(episode),
            "post_target_update": int(agent.training_step),
            "goal_index": int(goal_index),
            "post_target_goal": float(ladder[goal_index]),
            "consecutive_validation_passes": int(consecutive_passes),
            "source_checkpoint": os.path.abspath(checkpoint_path),
        }

    print("\n" + "=" * 70)
    print("Post-target CTDE-PPO 续训开始")
    print("=" * 70)
    print(f"源检查点={os.path.abspath(checkpoint_path)}")
    print(f"恢复post-target状态={resuming_post_target} | 起始回合={start_episode}")
    print(f"目标阶梯={ladder} | 当前目标={ladder[goal_index]:.2f}")
    print(f"学习率模式={config['lr_adapt_mode']}")

    training_log = {
        "phase": "post_target",
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "lr_adapt_mode": config["lr_adapt_mode"],
        "records": [],
        "final_update": None,
    }
    validation_log = {"phase": "post_target", "records": []}
    candidates = []
    update_info = {}
    completed_episode = start_episode - 1

    for episode in range(start_episode, config["post_target_episodes"] + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        coverage_history = []
        while not done:
            roles = agent.select_roles_if_required(obs, deterministic=False)
            if roles is not None:
                obs = env.apply_joint_role_assignment(roles)
            local_obs = obs["local_obs"]
            global_state = obs["global_state"]
            action_masks = obs["action_masks"]
            actions, log_probs = agent.select_actions(local_obs, action_masks)
            next_obs, rewards, done, info = env.step(actions)
            agent.accumulate_role_reward(rewards)
            if done or bool(next_obs.get("role_decision_required", 0)):
                agent.finish_role_option(next_obs, info)
            agent.store_transition(
                local_obs,
                global_state,
                actions,
                log_probs,
                rewards,
                done,
                next_global_state=next_obs["global_state"],
                truncated=bool(info.get("truncated", False)),
                action_masks=action_masks,
            )
            obs = next_obs
            episode_reward += float(sum(rewards))
            coverage_history.append(
                float(info.get("objective_coverage", info["boundary_coverage"]))
            )

        if len(agent.buffer) >= config["batch_size"]:
            settings = _post_target_optimizer_settings(agent.training_step, config)
            if settings["override_actor_lr"]:
                agent._set_actor_lr(settings["actor_lr"])
            agent._set_critic_lr(settings["critic_lr"])
            agent.clip_epsilon = settings["clip_epsilon"]
            agent.max_grad_norm = settings["max_grad_norm"]
            update_info = agent.update()
        role_update_info = agent.update_roles()

        metrics = _full_horizon_episode_metrics(
            coverage_history,
            info,
            config["stage3_success_target"],
            thresholds,
            [],
        )
        reward_breakdown = info.get("reward_breakdown") or {}
        training_log["records"].append(
            {
                "episode": int(episode),
                "phase": "post_target",
                "post_target_goal": float(ladder[goal_index]),
                "reward": episode_reward,
                "length": int(info["step"]),
                "done_reason": info["done_reason"],
                "scene_id": int(info["scene_id"]),
                "scene_key": str(info.get("scene_key", info["scene_id"])),
                "role_actor_loss": role_update_info.get("role_actor_loss", 0.0),
                "role_critic_loss": role_update_info.get("role_critic_loss", 0.0),
                "role_entropy": role_update_info.get("role_entropy", 0.0),
                "first_boundary_step": int(info.get("first_boundary_step", -1)),
                "reward_breakdown": reward_breakdown,
                "milestone_reward": float(reward_breakdown.get("r_milestone", 0.0)),
                "hold_reward": float(reward_breakdown.get("r_hold", 0.0)),
                "tail_reward": float(reward_breakdown.get("r_tail", 0.0)),
                "ppo_update": int(agent.training_step),
                "actor_lr": float(agent.actor_optimizer.param_groups[0]["lr"]),
                "critic_lr": float(agent.critic_optimizer.param_groups[0]["lr"]),
                "approx_kl": float(update_info.get("approx_kl", 0.0)),
                "kl_ema": float(update_info.get("kl_ema", 0.0)),
                "kl_lr_action": str(update_info.get("kl_lr_action", "fixed")),
                "clip_epsilon": float(agent.clip_epsilon),
                "max_grad_norm": float(agent.max_grad_norm),
                **{key: value for key, value in metrics.items() if key != "coverage_curve"},
            }
        )
        completed_episode = episode

        if episode % config["validation_interval"] == 0:
            eval_config = make_eval_config(
                config,
                config["validation_split"],
                config["validation_episodes_per_scene"],
                [3],
                evaluation_mode="full_horizon",
            )
            summary = _stage_summary(evaluate_preserving_rng(agent, eval_config), 3)
            previous_goal = ladder[goal_index]
            goal_index, consecutive_passes, advanced = _advance_post_target_goal(
                goal_index,
                consecutive_passes,
                summary,
                ladder,
                config["post_target_validation_patience"],
            )
            if advanced:
                env.set_post_target_goal(ladder[goal_index])
            validation_log["records"].append(
                {
                    "episode": int(episode),
                    "goal_before_validation": float(previous_goal),
                    "goal_after_validation": float(ladder[goal_index]),
                    "gate_passed": bool(_post_target_gate_passed(summary, previous_goal)),
                    "goal_advanced": bool(advanced),
                    "consecutive_validation_passes": int(consecutive_passes),
                    "summary": summary,
                }
            )

        if episode % config["save_interval"] == 0:
            path = os.path.join(model_dir, f"ppo_post_target_ep{episode:04d}.pth")
            agent.save(path, training_state(episode))
            candidates.append((f"ep{episode:04d}", path))

        if episode % config["log_interval"] == 0:
            recent = training_log["records"][-config["log_interval"]:]
            print(
                f"Episode {episode}/{config['post_target_episodes']} | "
                f"goal={ladder[goal_index]:.2f} | "
                f"tail100={np.mean([r['tail100_mean_coverage'] for r in recent]):.3f} | "
                f"AUC={np.mean([r['current_coverage_auc'] for r in recent]):.3f} | "
                f"updates={agent.training_step}"
            )

        if (
            config["max_train_updates"] is not None
            and agent.training_step >= config["max_train_updates"]
        ):
            break

    if len(agent.buffer) >= agent.min_update_batch_size:
        settings = _post_target_optimizer_settings(agent.training_step, config)
        if settings["override_actor_lr"]:
            agent._set_actor_lr(settings["actor_lr"])
        agent._set_critic_lr(settings["critic_lr"])
        agent.clip_epsilon = settings["clip_epsilon"]
        agent.max_grad_norm = settings["max_grad_norm"]
        update_info = agent.update(force=True)
        training_log["final_update"] = dict(update_info)
    else:
        agent.buffer.clear()
    role_final_update = agent.update_roles(force=True)
    if training_log["final_update"] is None:
        training_log["final_update"] = {}
    training_log["final_update"].update(role_final_update)

    final_path = os.path.join(model_dir, "ppo_post_target_final.pth")
    agent.save(final_path, training_state(completed_episode))
    if candidates and completed_episode % config["save_interval"] == 0:
        candidates[-1] = ("final", final_path)
    else:
        candidates.append(("final", final_path))

    _write_json(os.path.join(log_dir, "post_target_training_log.json"), training_log)
    _write_json(os.path.join(log_dir, "post_target_validation_log.json"), validation_log)

    candidate_results = []
    eligible = []
    for name, path in candidates:
        candidate_agent = _make_eval_agent(config, env)
        candidate_agent.load(path, restore_training_state=False)
        eval_config = make_eval_config(
            config,
            config["validation_split"],
            config["full_horizon_validation_episodes_per_scene"],
            [3],
            evaluation_mode="full_horizon",
        )
        summary = _stage_summary(evaluate_preserving_rng(candidate_agent, eval_config), 3)
        is_eligible = (
            float(summary.get("boundary_found_rate", 0.0)) >= 0.95
            and float(summary.get("threshold_reach_rate", {}).get("0.60", 0.0)) >= 0.60
            and float(summary.get("battery_depleted_rate", 0.0)) == 0.0
        )
        entry = {"name": name, "path": path, "eligible": is_eligible, "validation": summary}
        candidate_results.append(entry)
        if is_eligible:
            eligible.append(entry)

    source_path = os.path.abspath(checkpoint_path)
    best_path = None
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                float(item["validation"].get("mean_tail100_coverage", 0.0)),
                float(item["validation"].get("mean_current_coverage_auc", 0.0)),
                float(item["validation"].get("mean_historical_boundary_union_coverage", 0.0)),
            ),
        )
        best_path = os.path.join(model_dir, "ppo_best_post_target.pth")
        shutil.copy2(selected["path"], best_path)
        selected_path = best_path
        selection_status = "promoted"
    else:
        selected = {"name": "source_checkpoint", "path": source_path, "validation": None}
        selected_path = source_path
        selection_status = "source_retained_no_eligible_candidate"

    selected_agent = _make_eval_agent(config, env)
    selected_agent.load(selected_path, restore_training_state=False)
    split_results = {}
    for split in config["full_horizon_eval_splits"]:
        eval_config = make_eval_config(
            config,
            split,
            config["full_horizon_final_episodes_per_scene"],
            [3],
            evaluation_mode="full_horizon",
        )
        split_results[split] = _stage_summary(
            evaluate_preserving_rng(selected_agent, eval_config), 3
        )

    eval_summary = {
        "phase": "post_target",
        "source_checkpoint": source_path,
        "selection_status": selection_status,
        "selected": selected,
        "selected_model_path": selected_path,
        "candidates": candidate_results,
        "splits": split_results,
    }
    _write_json(os.path.join(log_dir, "post_target_eval_summary.json"), eval_summary)
    _save_source_snapshot(output_dir)
    print(f"Post-target训练完成 | selected={selected_path}")
    return agent, training_log, validation_log, eval_summary, output_dir, best_path


def evaluate(agent: CTDE_PPO_Agent, config: Dict, num_episodes: int = None) -> Dict:
    config = normalize_training_config(config)
    _resolve_dataset_scene_keys(config)
    eval_episodes_per_scene = int(config["eval_episodes_per_scene"])
    evaluation_mode = str(config["evaluation_mode"])
    full_horizon = evaluation_mode == "full_horizon"
    curve_stride = int(config["full_horizon_curve_stride"])
    coverage_thresholds = list(config["full_horizon_thresholds"])
    if num_episodes is not None:
        eval_episodes_per_scene = max(1, int(num_episodes) // max(len(config["eval_scene_keys"]), 1))

    actor_was_training = agent.actor.training
    critic_was_training = agent.critic.training
    role_actor_was_training = (
        agent.role_agent.actor.training if agent.role_agent is not None else None
    )
    role_critic_was_training = (
        agent.role_agent.critic.training if agent.role_agent is not None else None
    )
    agent.actor.eval()
    agent.critic.eval()
    if agent.role_agent is not None:
        agent.role_agent.actor.eval()
        agent.role_agent.critic.eval()

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
                    communication_enabled=config["communication_enabled"],
                    communication_radius_factor=config["communication_radius_factor"],
                    action_mask_enabled=config["action_mask_enabled"],
                    novelty_reward_weight=config["novelty_reward_weight"],
                    novelty_step_penalty=config["novelty_step_penalty"],
                    novelty_revisit_penalty=config["novelty_revisit_penalty"],
                    invalid_action_penalty=config["invalid_action_penalty"],
                    team_overlap_penalty=config["team_overlap_penalty"],
                    curriculum_stage=int(stage),
                    mode=config["eval_split"],
                    fixed_scene_key=str(scene_key),
                    init_percentile=config["init_percentile"],
                    init_area_percent=config["init_area_percent"],
                    stage2_target=config["stage2_success_target"],
                    stage3_target=config["stage3_success_target"],
                    stage3_near_prob=0.0,
                    termination_mode=evaluation_mode,
                    **_persistent_env_kwargs(config),
                )
                env.set_curriculum_substage(str(config.get("curriculum_substage", stage)))

                for episode_index in range(eval_episodes_per_scene):
                    obs = env.reset()
                    done = False
                    episode_reward = 0.0
                    episode_length = 0
                    coverage_history = []
                    coverage_curve = [
                        {
                            "step": 0,
                            "coverage": 0.0,
                            "historical_boundary_union_coverage": 0.0,
                        }
                    ] if full_horizon else []
                    while not done:
                        if int(stage) < 3 and bool(obs.get("role_decision_required", 0)):
                            roles = env.heuristic_joint_role_assignment()
                        else:
                            roles = agent.select_roles_if_required(
                                obs, deterministic=True, track_option=False
                            )
                        if roles is not None:
                            obs = env.apply_joint_role_assignment(roles)
                        actions = agent.select_actions_deterministic(
                            obs["local_obs"], obs["action_masks"]
                        )
                        obs, rewards, done, info = env.step(actions)
                        episode_reward += float(sum(rewards))
                        episode_length += 1
                        if full_horizon:
                            coverage_history.append(
                                float(
                                    info.get(
                                        "objective_coverage",
                                        info["boundary_coverage"],
                                    )
                                )
                            )
                            if (
                                episode_length % curve_stride == 0
                                or info["boundary_refreshed"]
                                or done
                            ):
                                if coverage_curve[-1]["step"] != episode_length:
                                    coverage_curve.append(
                                        {
                                            "step": int(episode_length),
                                            "coverage": float(
                                                info.get(
                                                    "objective_coverage",
                                                    info["boundary_coverage"],
                                                )
                                            ),
                                            "exact_boundary_coverage": float(
                                                info["boundary_coverage"]
                                            ),
                                            "fresh_boundary_coverage": float(
                                                info.get("fresh_boundary_coverage", 0.0)
                                            ),
                                            "tolerant_boundary_coverage": float(
                                                info.get("tolerant_boundary_coverage", 0.0)
                                            ),
                                            "historical_boundary_union_coverage": float(
                                                info["historical_boundary_union_coverage"]
                                            ),
                                            "boundary_refreshed": bool(info["boundary_refreshed"]),
                                            "coverage_before_refresh": float(
                                                info["coverage_before_boundary_refresh"]
                                            ),
                                            "coverage_after_refresh": float(
                                                info["coverage_after_boundary_refresh"]
                                            ),
                                            "coverage_action_gain": float(
                                                info.get("coverage_action_gain", 0.0)
                                            ),
                                            "coverage_refresh_drop": float(
                                                info.get("coverage_refresh_drop", 0.0)
                                            ),
                                        }
                                    )

                    success = (
                        bool(info.get("target_reached", False))
                        if full_horizon
                        else info["done_reason"] == "mission_complete"
                    )
                    record = {
                        "scene_id": int(info.get("scene_id", -1)),
                        "scene_key": str(info.get("scene_key", scene_key)),
                        "episode_index": int(episode_index),
                        "eval_seed": int(stage_seed),
                        "evaluation_mode": evaluation_mode,
                        "observation_profile": config["observation_profile"],
                        "reward_profile": config["reward_profile"],
                        "reward": episode_reward,
                        "coverage": float(
                            info.get("objective_coverage", info["boundary_coverage"])
                        ),
                        "exact_boundary_coverage": float(info["boundary_coverage"]),
                        "fresh_boundary_coverage": float(
                            info.get("fresh_boundary_coverage", 0.0)
                        ),
                        "tolerant_boundary_coverage": float(
                            info.get("tolerant_boundary_coverage", 0.0)
                        ),
                        "success": 1.0 if success else 0.0,
                        "length": int(episode_length),
                        "timeout": 1.0 if info["done_reason"] == "max_steps_reached" else 0.0,
                        "zero_coverage_timeout": 1.0 if info.get("zero_coverage_timeout", False) else 0.0,
                        "zero_discovery_timeout": 1.0
                        if info.get("zero_discovery_timeout", False)
                        else 0.0,
                        "done_reason": info["done_reason"],
                        "first_heat_step": int(info.get("first_heat_step", -1)),
                        "first_boundary_step": int(info.get("first_boundary_step", -1)),
                        "stage1_tracking_success": bool(
                            info.get("stage1_tracking_success", False)
                        ),
                        "stage1_unique_boundary_cells": int(
                            info.get("stage1_unique_boundary_cells", 0)
                        ),
                        "major_refresh_count": int(info.get("major_refresh_count", 0)),
                        "refresh_recovery_successes": int(
                            info.get("refresh_recovery_successes", 0)
                        ),
                        "mean_refresh_recovery_time": info.get(
                            "mean_refresh_recovery_time"
                        ),
                        "pre_boundary_revisit_ratio": float(
                            info.get("pre_boundary_revisit_ratio", 0.0)
                        ),
                        "team_overlap_ratio": float(info.get("team_overlap_ratio", 0.0)),
                        "communication_available_rate": float(
                            info.get("communication_available_rate", 0.0)
                        ),
                        "invalid_action_count": int(info.get("invalid_action_count", 0)),
                        "role_switch_count": int(info.get("role_switch_count", 0)),
                        "communication_message_expirations": int(
                            info.get("communication_message_expirations", 0)
                        ),
                        "boundary_report_expirations": int(
                            info.get("boundary_report_expirations", 0)
                        ),
                        "reward_breakdown": info.get("reward_breakdown") or {},
                    }
                    if full_horizon:
                        target = (
                            config["stage2_success_target"]
                            if int(stage) == 2
                            else config["stage3_success_target"]
                        )
                        record.update(
                            _full_horizon_episode_metrics(
                                coverage_history,
                                info,
                                target,
                                coverage_thresholds,
                                coverage_curve,
                            )
                        )
                    else:
                        record["task_score"] = _task_score(
                            info.get("objective_coverage", info["boundary_coverage"]),
                            success,
                            episode_length,
                            config["max_steps"],
                        )
                    stage_records.append(record)

            if full_horizon:
                target = (
                    config["stage2_success_target"]
                    if int(stage) == 2
                    else config["stage3_success_target"]
                )
                results[int(stage)] = _summarize_full_horizon_records(stage_records, target)
            else:
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
                    "zero_discovery_timeout_rate": float(
                        np.mean([r["zero_discovery_timeout"] for r in stage_records])
                    ),
                    "boundary_found_rate": float(
                        np.mean(
                            [int(r.get("first_boundary_step", -1)) > 0 for r in stage_records]
                        )
                    ),
                    "stable_tracking_success_rate": float(
                        np.mean([r["stage1_tracking_success"] for r in stage_records])
                    ),
                    "median_first_boundary_step": float(
                        np.median(
                            [
                                r["first_boundary_step"]
                                for r in stage_records
                                if r["first_boundary_step"] > 0
                            ]
                        )
                    ) if any(r["first_boundary_step"] > 0 for r in stage_records) else None,
                    "median_unique_boundary_cells": float(
                        np.median(
                            [r["stage1_unique_boundary_cells"] for r in stage_records]
                        )
                    ),
                    "mean_pre_boundary_revisit_ratio": float(
                        np.mean([r["pre_boundary_revisit_ratio"] for r in stage_records])
                    ),
                    "mean_team_overlap_ratio": float(
                        np.mean([r["team_overlap_ratio"] for r in stage_records])
                    ),
                    "mean_communication_available_rate": float(
                        np.mean([r["communication_available_rate"] for r in stage_records])
                    ),
                    "mean_invalid_action_count": float(
                        np.mean([r["invalid_action_count"] for r in stage_records])
                    ),
                    "mean_task_score": float(np.mean([r["task_score"] for r in stage_records])),
                    "records": stage_records,
                }

            summary = results[int(stage)]
            if full_horizon:
                print(
                    f"完整时域评估阶段={stage} | 回合数={summary['episodes']} | "
                    f"覆盖AUC={summary['mean_current_coverage_auc'] * 100:.1f}% | "
                    f"末100步覆盖={summary['mean_tail100_coverage'] * 100:.1f}% | "
                    f"60%到达率={summary['target_reach_rate'] * 100:.1f}%"
                )
            else:
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
        if agent.role_agent is not None:
            agent.role_agent.actor.train(role_actor_was_training)
            agent.role_agent.critic.train(role_critic_was_training)

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

            full_horizon_summary_path = os.path.join(
                output_dir,
                "logs",
                "full_horizon_summary.json",
            )
            full_horizon_summary = {}
            if os.path.exists(full_horizon_summary_path):
                with open(full_horizon_summary_path, "r", encoding="utf-8") as f:
                    full_horizon_summary = json.load(f)

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
                "full_horizon_summary_path": full_horizon_summary_path,
                "full_horizon_summary": full_horizon_summary,
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
    parser.add_argument("--init-percentile", type=float, default=None)
    parser.add_argument("--init-area-percent", type=float, default=None)
    parser.add_argument("--use-metadata-uav-params", action="store_true")
    parser.add_argument("--observation-profile", type=str, default=None)
    parser.add_argument("--reward-profile", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-full-horizon-eval", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--post-target-train", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--post-target-episodes", type=int, default=None)
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
    if not args.single_train and not args.post_target_train:
        seed_mode = _prompt_seed_mode()
        config["comparison_seeds"] = [42] if seed_mode == "single" else [42, 43, 44]
    if args.lr_mode is not None:
        config["lr_adapt_mode"] = args.lr_mode
    if args.target_kl is not None:
        config["target_kl"] = args.target_kl
        config["quality_target_kl"] = args.target_kl
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
        config["full_horizon_eval_after_train"] = False
    if args.no_full_horizon_eval:
        config["full_horizon_eval_after_train"] = False
    if args.no_plot:
        config["plot_after_train"] = False

    if args.post_target_train:
        config["post_target_train"] = True
        config["post_target_resume_checkpoint"] = args.resume_checkpoint
        if args.post_target_episodes is not None:
            config["post_target_episodes"] = args.post_target_episodes
        train_post_target(config, args.resume_checkpoint)
        return

    if args.single_train:
        train(config)
    else:
        run_lr_comparison(config)


if __name__ == "__main__":
    main()
