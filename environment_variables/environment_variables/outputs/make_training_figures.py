"""Generate training figures from saved CTDE-PPO baseline logs.

Current training output layout:
    outputs/<timestamp>/training_results/logs/training_log.npz
    outputs/<timestamp>/training_results/logs/model_quality_metrics.json

LR comparison layout:
    outputs/lr_comparison_<timestamp>/training_results/<variant>/logs/training_log.npz

Default behavior:
    python make_training_figures.py

The script finds the latest compatible result directory and writes figures to:
    outputs/<timestamp>/figures/training_figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR_NAME = "training_results"
LOGS_DIR_NAME = "logs"
TRAINING_LOG_NAME = "training_log.npz"
QUALITY_METRICS_NAME = "model_quality_metrics.json"
BASELINE_NAME = "Baseline_CTDE_PPO"

RUN_ORDER = [
    BASELINE_NAME,
    "Fixed_LR_CTDE_PPO",
    "KL_LR_CTDE_PPO",
    "Full_SDF_Infotaxis",
    "No_SDF",
    "No_Infotaxis",
    "Vanilla_CTDE_PPO",
]

LABELS = {
    BASELINE_NAME: "Baseline CTDE-PPO",
    "Fixed_LR_CTDE_PPO": "Fixed LR CTDE-PPO",
    "KL_LR_CTDE_PPO": "KL-adaptive LR CTDE-PPO",
    "Full_SDF_Infotaxis": "Full (SDF + Infotaxis)",
    "No_SDF": "No SDF",
    "No_Infotaxis": "No Infotaxis",
    "Vanilla_CTDE_PPO": "Vanilla CTDE-PPO",
}

COLORS = {
    BASELINE_NAME: "#1f77b4",
    "Fixed_LR_CTDE_PPO": "#4c78a8",
    "KL_LR_CTDE_PPO": "#f58518",
    "Full_SDF_Infotaxis": "#d62728",
    "No_SDF": "#1f77b4",
    "No_Infotaxis": "#2ca02c",
    "Vanilla_CTDE_PPO": "#9467bd",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Times New Roman",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_dir_for_log(log_path: Path) -> Path:
    if log_path.name.startswith("training_log_"):
        return log_path.parent

    output_dir = log_path.parent.parent
    if output_dir.parent.name == RESULTS_DIR_NAME:
        return output_dir.parent
    return output_dir


def collect_training_logs(result_dir: Path) -> List[Path]:
    paths: List[Path] = []

    direct = result_dir / LOGS_DIR_NAME / TRAINING_LOG_NAME
    if direct.is_file():
        paths.append(direct)

    paths.extend(sorted(result_dir.glob(f"*/{LOGS_DIR_NAME}/{TRAINING_LOG_NAME}")))
    paths.extend(sorted(result_dir.glob("training_log_*.npz")))

    if not paths:
        paths.extend(sorted(result_dir.glob(f"**/{LOGS_DIR_NAME}/{TRAINING_LOG_NAME}")))
        paths.extend(sorted(result_dir.glob("**/training_log_*.npz")))

    seen = set()
    unique_paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def has_training_logs(path: Path) -> bool:
    return path.is_dir() and bool(collect_training_logs(path))


def find_latest_results(outputs_dir: Path) -> Path:
    log_paths = list(outputs_dir.glob(f"**/{LOGS_DIR_NAME}/{TRAINING_LOG_NAME}"))
    log_paths.extend(outputs_dir.glob("**/training_log_*.npz"))
    if not log_paths:
        raise FileNotFoundError(f"No saved training logs found under {outputs_dir}")

    latest_log = max(log_paths, key=lambda p: p.stat().st_mtime)
    return result_dir_for_log(latest_log)


def resolve_results_dir(raw_dir: str | None, outputs_dir: Path) -> Path:
    if raw_dir is None:
        return find_latest_results(outputs_dir)

    candidate = Path(raw_dir)
    if not candidate.is_absolute():
        candidate = outputs_dir / candidate

    if (candidate / RESULTS_DIR_NAME).is_dir() and has_training_logs(candidate / RESULTS_DIR_NAME):
        return candidate / RESULTS_DIR_NAME

    if has_training_logs(candidate):
        return candidate

    nested_logs = collect_training_logs(candidate)
    if nested_logs:
        return result_dir_for_log(max(nested_logs, key=lambda p: p.stat().st_mtime))

    raise FileNotFoundError(f"No saved training logs found in {candidate}")


def default_out_dir(result_dir: Path) -> Path:
    if result_dir.name == RESULTS_DIR_NAME:
        return result_dir.parent / "figures" / "training_figures"
    if result_dir.parent.name == RESULTS_DIR_NAME:
        return result_dir / "training_figures"
    return result_dir / "training_figures"


def output_dir_for_log_path(path: Path) -> Path:
    if path.name.startswith("training_log_"):
        return path.parent
    return path.parent.parent


def infer_run_name(path: Path, config: Mapping) -> str:
    if path.name.startswith("training_log_"):
        return path.stem.replace("training_log_", "")

    output_dir = output_dir_for_log_path(path)
    configured = config.get("variant_name")
    if configured:
        if output_dir.name.startswith(f"{configured}_seed"):
            return output_dir.name
        return str(configured)

    if output_dir.name in {RESULTS_DIR_NAME, "outputs"}:
        return BASELINE_NAME
    return output_dir.name


def base_run_name(name: str) -> str:
    text = str(name)
    marker = "_seed"
    if marker in text:
        prefix, suffix = text.rsplit(marker, 1)
        if suffix.isdigit():
            return prefix
    return text


def ordered_names(names: Iterable[str]) -> List[str]:
    seen = list(names)
    ordered = [name for name in RUN_ORDER if name in seen]
    ordered.extend(sorted(name for name in seen if name not in ordered))
    return ordered


def load_logs(result_dir: Path):
    logs: Dict[str, Mapping[str, np.ndarray]] = {}
    configs: Dict[str, Dict] = {}
    metrics: Dict[str, Dict] = {}

    for path in collect_training_logs(result_dir):
        output_dir = output_dir_for_log_path(path)
        config = read_json(output_dir / "config.json")
        name = infer_run_name(path, config)
        base_name = name
        suffix = 2
        while name in logs:
            name = f"{base_name}_{suffix}"
            suffix += 1

        logs[name] = np.load(path, allow_pickle=True)
        configs[name] = config
        metrics[name] = read_json(output_dir / LOGS_DIR_NAME / QUALITY_METRICS_NAME)

    ordered_logs = {name: logs[name] for name in ordered_names(logs)}
    ordered_configs = {name: configs[name] for name in ordered_logs}
    ordered_metrics = {name: metrics[name] for name in ordered_logs}
    if not ordered_logs:
        raise RuntimeError(f"No logs loaded from {result_dir}")
    return ordered_logs, ordered_configs, ordered_metrics


def filter_runs(
    logs: Dict[str, Mapping[str, np.ndarray]],
    configs: Dict[str, Dict],
    metrics: Dict[str, Dict],
    run_filter: str | None,
):
    if not run_filter:
        return logs, configs, metrics
    kept = {name: log for name, log in logs.items() if run_filter in name}
    if not kept:
        raise RuntimeError(f"No runs matched filter: {run_filter}")
    return (
        kept,
        {name: configs[name] for name in kept},
        {name: metrics[name] for name in kept},
    )


def has_key(log: Mapping[str, np.ndarray], key: str) -> bool:
    return key in log


def get_array(log: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in log:
        raise KeyError(f"missing key in training log: {key}")
    return np.asarray(log[key])


def float_array(log: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    return get_array(log, key).astype(float)


def episodes_for(log: Mapping[str, np.ndarray], fallback_key: str = "rewards") -> np.ndarray:
    if "episodes" in log:
        return get_array(log, "episodes").astype(float)
    return np.arange(1, len(get_array(log, fallback_key)) + 1, dtype=float)


def max_steps_for(configs: Mapping[str, Dict], override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    values = [int(cfg.get("max_steps", 0)) for cfg in configs.values() if cfg.get("max_steps")]
    return max(values) if values else 600


def target_kl_for(configs: Mapping[str, Dict]) -> float:
    values = [float(cfg.get("target_kl")) for cfg in configs.values() if cfg.get("target_kl") is not None]
    return values[0] if values else 0.015


def effective_window(length: int, window: int) -> int:
    if length <= 0 or window <= 1:
        return 1
    return min(window, max(1, length // 3), length)


def rolling_mean(values: Iterable[float], window: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])
    w = effective_window(arr.size, window)
    if w <= 1:
        return np.arange(1, arr.size + 1, dtype=float), arr
    kernel = np.ones(w, dtype=float) / w
    return np.arange(w, arr.size + 1, dtype=float), np.convolve(arr, kernel, mode="valid")


def save_current(out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def label_for(name: str) -> str:
    base = base_run_name(name)
    return LABELS.get(base, base)


def color_for(name: str):
    return COLORS.get(base_run_name(name), None)


def raw_alpha(length: int) -> float:
    return 0.12 if length <= 2000 else 0.06


def task_scores(log: Mapping[str, np.ndarray], max_steps: int) -> np.ndarray:
    if has_key(log, "task_scores"):
        return float_array(log, "task_scores") * 100.0
    coverage = float_array(log, "coverages")
    success = float_array(log, "success")
    length = float_array(log, "lengths")
    efficiency = success * (1.0 - np.clip(length / max(float(max_steps), 1.0), 0.0, 1.0))
    return 100.0 * (0.5 * coverage + 0.3 * success + 0.2 * efficiency)


def grouped_by_base(logs: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, List[Tuple[str, Mapping[str, np.ndarray]]]]:
    grouped: Dict[str, List[Tuple[str, Mapping[str, np.ndarray]]]] = {}
    for name, log in logs.items():
        grouped.setdefault(base_run_name(name), []).append((name, log))
    return {name: grouped[name] for name in ordered_names(grouped)}


def aligned_mean_std(series: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    series = [np.asarray(values, dtype=float) for values in series if np.asarray(values).size]
    if not series:
        return np.asarray([]), np.asarray([])
    n = min(len(values) for values in series)
    if n <= 0:
        return np.asarray([]), np.asarray([])
    arr = np.vstack([values[:n] for values in series])
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def values_for_metric(log: Mapping[str, np.ndarray], metric_name: str, max_steps: int) -> np.ndarray:
    if metric_name == "task_score":
        return task_scores(log, max_steps)
    if metric_name == "coverage" and has_key(log, "coverages"):
        return float_array(log, "coverages") * 100.0
    if metric_name == "success" and has_key(log, "success"):
        return float_array(log, "success") * 100.0
    if metric_name == "timeout" and has_key(log, "timeout"):
        return float_array(log, "timeout") * 100.0
    if metric_name == "zero_coverage_timeout" and has_key(log, "zero_coverage_timeout"):
        return float_array(log, "zero_coverage_timeout") * 100.0
    if metric_name == "steps" and has_key(log, "lengths"):
        return float_array(log, "lengths")
    if has_key(log, metric_name):
        return float_array(log, metric_name)
    return np.asarray([], dtype=float)


def plot_aggregate_metric(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    metric_name: str,
    ylabel: str,
    title: str,
    out_path: Path,
    window: int,
    dpi: int,
    max_steps: int,
    ylim: Tuple[float, float] | None = None,
) -> bool:
    plt.figure(figsize=(9.0, 5.2))
    plotted = False
    for base, items in grouped_by_base(logs).items():
        xs = []
        ys = []
        for _, log in items:
            values = values_for_metric(log, metric_name, max_steps)
            if values.size == 0:
                continue
            episodes = episodes_for(log, "rewards")
            n = min(len(values), len(episodes))
            x_smooth, y_smooth = rolling_mean(values[:n], window)
            if x_smooth.size:
                xs.append(episodes[x_smooth.astype(int) - 1])
                ys.append(y_smooth)
        mean, std = aligned_mean_std(ys)
        if mean.size:
            x = xs[0][: len(mean)]
            color = color_for(base)
            plt.plot(x, mean, color=color, linewidth=2.0, label=label_for(base))
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_aggregate_two_panel(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    panels: List[Tuple[str, str, str]],
    out_path: Path,
    window: int,
    dpi: int,
    max_steps: int,
    ylims: List[Tuple[float, float] | None] | None = None,
) -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    ylims = ylims or [None, None]
    for ax, (metric_name, ylabel, title), ylim in zip(axes, panels, ylims):
        for base, items in grouped_by_base(logs).items():
            xs = []
            ys = []
            for _, log in items:
                values = values_for_metric(log, metric_name, max_steps)
                if values.size == 0:
                    continue
                episodes = episodes_for(log, "rewards")
                n = min(len(values), len(episodes))
                x_smooth, y_smooth = rolling_mean(values[:n], window)
                if x_smooth.size:
                    xs.append(episodes[x_smooth.astype(int) - 1])
                    ys.append(y_smooth)
            mean, std = aligned_mean_std(ys)
            if mean.size:
                x = xs[0][: len(mean)]
                color = color_for(base)
                ax.plot(x, mean, color=color, linewidth=2.0, label=label_for(base))
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
                plotted = True
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def aggregate_scalar_by_base(values_by_run: Mapping[str, float]) -> Tuple[List[str], List[float], List[float]]:
    names = ordered_names({base_run_name(name) for name in values_by_run})
    means = []
    stds = []
    for base in names:
        vals = [value for name, value in values_by_run.items() if base_run_name(name) == base and np.isfinite(value)]
        means.append(float(np.nanmean(vals)) if vals else np.nan)
        stds.append(float(np.nanstd(vals)) if vals else np.nan)
    return names, means, stds


def plot_aggregate_last_window_summary(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    dpi: int,
    max_steps: int,
    last_n: int = 100,
) -> bool:
    metrics = [
        ("reward", "Mean Reward"),
        ("task_score", "Task Score (%)"),
        ("coverage", "Coverage (%)"),
        ("success", "Success Rate (%)"),
        ("steps", "Episode Steps"),
        ("timeout", "Timeout Rate (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    plotted = False
    for ax, (metric_name, title) in zip(axes.ravel(), metrics):
        run_vals = {}
        for name, log in logs.items():
            values = values_for_metric(log, metric_name, max_steps)
            run_vals[name] = float(np.nanmean(values[-last_n:])) if values.size else np.nan
        names, means, stds = aggregate_scalar_by_base(run_vals)
        if np.all(np.isnan(means)):
            continue
        x = np.arange(len(names))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=[color_for(name) or "#777777" for name in names],
            alpha=0.86,
        )
        ax.set_title(f"Last {last_n}: {title}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(name) for name in names], rotation=18, ha="right")
        ax.set_ylabel(title)
        if title.endswith("(%)"):
            ax.set_ylim(0, 105)
        for bar, val in zip(bars, means):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        plotted = True
    if not plotted:
        plt.close()
        return False
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_aggregate_quality_summary(metrics: Mapping[str, Dict], out_path: Path, dpi: int) -> bool:
    if not any(metrics.values()):
        return False
    specs = [
        (["convergence_efficiency", "auc_task_score_by_steps"], "AUC Task Score (%)", 100.0),
        (["convergence_efficiency", "steps_to_threshold"], "Steps to Threshold", 1.0),
        (["reward_stability", "task_score_std_tail"], "Tail Task Std (%)", 100.0),
        (["kl_stability", "kl_overshoot_rate"], "KL Overshoot Rate (%)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    plotted = False
    for ax, (path, title, scale) in zip(axes.ravel(), specs):
        run_vals = {}
        for name, data in metrics.items():
            value = metric_value(data, path)
            run_vals[name] = float(value) * scale if value is not None else np.nan
        names, means, stds = aggregate_scalar_by_base(run_vals)
        if np.all(np.isnan(means)):
            continue
        x = np.arange(len(names))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=[color_for(name) or "#777777" for name in names],
            alpha=0.86,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(name) for name in names], rotation=18, ha="right")
        ax.set_ylabel(title)
        for bar, val in zip(bars, means):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2g}", ha="center", va="bottom", fontsize=8)
        plotted = True
    if not plotted:
        plt.close()
        return False
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_smoothed_metric(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    window: int,
    dpi: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ylim: Tuple[float, float] | None = None,
    raw: bool = False,
) -> bool:
    plt.figure(figsize=(9.0, 5.2))
    plotted = False

    for name, log in logs.items():
        if not has_key(log, key):
            continue
        values = float_array(log, key)
        if transform is not None:
            values = transform(values)

        episodes = episodes_for(log, key)
        n = min(len(values), len(episodes))
        values = values[:n]
        episodes = episodes[:n]
        if n == 0:
            continue

        if raw:
            plt.plot(episodes, values, color=color_for(name), alpha=raw_alpha(n), linewidth=0.7)

        x_smooth, y_smooth = rolling_mean(values, window)
        if x_smooth.size:
            smooth_episodes = episodes[x_smooth.astype(int) - 1]
            plt.plot(smooth_episodes, y_smooth, color=color_for(name), linewidth=1.8, label=label_for(name))
            plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_task_score(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    window: int,
    dpi: int,
    max_steps: int,
) -> bool:
    plt.figure(figsize=(9.0, 5.2))
    plotted = False
    for name, log in logs.items():
        if not (has_key(log, "task_scores") or all(has_key(log, key) for key in ["coverages", "success", "lengths"])):
            continue
        values = task_scores(log, max_steps)
        episodes = episodes_for(log, "rewards")
        n = min(len(values), len(episodes))
        x_smooth, y_smooth = rolling_mean(values[:n], window)
        if x_smooth.size:
            plt.plot(episodes[x_smooth.astype(int) - 1], y_smooth, color=color_for(name), linewidth=1.9, label=label_for(name))
            plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title("Training Task Score", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Task Score (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_stage_curve(logs: Mapping[str, Mapping[str, np.ndarray]], out_path: Path, dpi: int) -> bool:
    plt.figure(figsize=(9.0, 4.8))
    plotted = False
    for name, log in logs.items():
        if not has_key(log, "stage"):
            continue
        stage = float_array(log, "stage")
        episodes = episodes_for(log, "stage")
        n = min(len(stage), len(episodes))
        if n == 0:
            continue
        plt.step(
            episodes[:n],
            stage[:n],
            where="post",
            color=color_for(name),
            label=label_for(name),
            linewidth=2.0,
            alpha=0.9,
        )
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title("Curriculum Stage Transition", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Stage", fontsize=12)
    plt.yticks([1, 2, 3], ["Stage 1", "Stage 2", "Stage 3"])
    plt.ylim(0.85, 3.15)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_done_reasons(logs: Mapping[str, Mapping[str, np.ndarray]], out_path: Path, dpi: int, last_n: int = 100) -> bool:
    counters: Dict[str, Counter[str]] = {}
    for name, log in logs.items():
        if not has_key(log, "done_reasons"):
            continue
        reasons = np.asarray(get_array(log, "done_reasons")).astype(str)
        if last_n > 0:
            reasons = reasons[-last_n:]
        counters[name] = Counter(reasons.tolist())

    if not counters:
        return False

    preferred = ["mission_complete", "max_steps_reached", "battery_depleted", "other"]
    all_reasons = {reason for counter in counters.values() for reason in counter}
    categories = [reason for reason in preferred if reason in all_reasons]
    categories.extend(sorted(reason for reason in all_reasons if reason not in categories))

    names = list(counters.keys())
    x = np.arange(len(names))
    bottom = np.zeros(len(names), dtype=float)
    palette = {
        "mission_complete": "#2ca02c",
        "max_steps_reached": "#d62728",
        "battery_depleted": "#7f7f7f",
        "other": "#ff7f0e",
    }

    plt.figure(figsize=(9.2, 5.2))
    for category in categories:
        vals = []
        for name in names:
            total = max(1, sum(counters[name].values()))
            vals.append(counters[name].get(category, 0) / total * 100.0)
        vals_arr = np.asarray(vals)
        plt.bar(x, vals_arr, bottom=bottom, label=category, color=palette.get(category, None), alpha=0.88)
        bottom += vals_arr

    plt.title(f"Done Reason Distribution (Last {last_n} Episodes)", fontsize=14, fontweight="bold")
    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(x, [label_for(name) for name in names], rotation=18, ha="right")
    plt.ylim(0, 100)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_loss_curves(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    window: int,
    dpi: int,
) -> bool:
    if not any(has_key(log, "actor_loss") or has_key(log, "critic_loss") for log in logs.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    for ax, key, ylabel, title in [
        (axes[0], "actor_loss", "Actor Loss", "Actor Loss"),
        (axes[1], "critic_loss", "Critic Loss", "Critic Loss"),
    ]:
        for name, log in logs.items():
            if not has_key(log, key):
                continue
            values = float_array(log, key)
            episodes = episodes_for(log, key)
            n = min(len(values), len(episodes))
            x_smooth, y_smooth = rolling_mean(values[:n], window)
            if x_smooth.size:
                ax.plot(episodes[x_smooth.astype(int) - 1], y_smooth, color=color_for(name), linewidth=1.8, label=label_for(name))
                plotted = True
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def unique_update_series(log: Mapping[str, np.ndarray], key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not (has_key(log, "ppo_updates") and has_key(log, key)):
        return np.array([]), np.array([])
    updates = get_array(log, "ppo_updates").astype(int)
    values = float_array(log, key)
    seen = set()
    xs = []
    ys = []
    for update_id, value in zip(updates, values):
        update_id = int(update_id)
        if update_id <= 0 or update_id in seen:
            continue
        seen.add(update_id)
        xs.append(update_id)
        ys.append(float(value))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_ppo_diagnostics(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    configs: Mapping[str, Dict],
    out_path: Path,
    dpi: int,
) -> bool:
    if not any(has_key(log, "approx_kl") or has_key(log, "clip_fraction") for log in logs.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    target_kl = target_kl_for(configs)

    for name, log in logs.items():
        x, y = unique_update_series(log, "approx_kl")
        if x.size:
            axes[0].plot(x, y, color=color_for(name), linewidth=1.8, label=f"{label_for(name)} KL")
            plotted = True
        x_ema, y_ema = unique_update_series(log, "kl_ema")
        if x_ema.size:
            axes[0].plot(x_ema, y_ema, color=color_for(name), linewidth=1.4, linestyle="--", alpha=0.85, label=f"{label_for(name)} KL EMA")
            plotted = True

        x_clip, y_clip = unique_update_series(log, "clip_fraction")
        if x_clip.size:
            axes[1].plot(x_clip, y_clip * 100.0, color=color_for(name), linewidth=1.8, label=label_for(name))
            plotted = True

    axes[0].axhline(target_kl, color="#444444", linestyle=":", linewidth=1.4, label=f"target KL={target_kl:g}")
    axes[0].set_title("PPO KL Diagnostics", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("PPO Update")
    axes[0].set_ylabel("KL")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("PPO Clip Fraction", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("PPO Update")
    axes[1].set_ylabel("Clip Fraction (%)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def plot_learning_rate(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    dpi: int,
) -> bool:
    if not any(has_key(log, "actor_lr") for log in logs.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    action_names = ["up", "down", "keep", "fixed"]
    action_counts = {}

    for name, log in logs.items():
        if has_key(log, "actor_lr"):
            values = float_array(log, "actor_lr")
            episodes = episodes_for(log, "actor_lr")
            n = min(len(values), len(episodes))
            if n:
                axes[0].plot(episodes[:n], values[:n], color=color_for(name), linewidth=1.8, label=label_for(name))
                plotted = True

        if has_key(log, "kl_lr_action"):
            actions = np.asarray(get_array(log, "kl_lr_action")).astype(str)
            action_counts[name] = Counter(actions.tolist())

    axes[0].set_title("Actor Learning Rate", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Actor LR")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)

    if action_counts:
        names = list(action_counts)
        x = np.arange(len(names))
        bottom = np.zeros(len(names), dtype=float)
        palette = {"up": "#2ca02c", "down": "#d62728", "keep": "#ffbf00", "fixed": "#7f7f7f"}
        for action in action_names:
            vals = np.asarray([action_counts[name].get(action, 0) for name in names], dtype=float)
            axes[1].bar(x, vals, bottom=bottom, color=palette[action], label=action, alpha=0.88)
            bottom += vals
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([label_for(name) for name in names], rotation=18, ha="right")
        axes[1].set_ylabel("Episode Count")
        axes[1].legend(frameon=False, fontsize=8)
        plotted = True

    axes[1].set_title("KL LR Action Counts", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Run")
    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def plot_timeout_rates(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    window: int,
    dpi: int,
) -> bool:
    if not any(has_key(log, "timeout") or has_key(log, "zero_coverage_timeout") for log in logs.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    for ax, key, title in [
        (axes[0], "timeout", "Timeout Rate"),
        (axes[1], "zero_coverage_timeout", "Zero-coverage Timeout Rate"),
    ]:
        for name, log in logs.items():
            if not has_key(log, key):
                continue
            values = float_array(log, key) * 100.0
            episodes = episodes_for(log, key)
            n = min(len(values), len(episodes))
            x_smooth, y_smooth = rolling_mean(values[:n], window)
            if x_smooth.size:
                ax.plot(episodes[x_smooth.astype(int) - 1], y_smooth, color=color_for(name), linewidth=1.8, label=label_for(name))
                plotted = True
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def plot_progress_budget(logs: Mapping[str, Mapping[str, np.ndarray]], out_path: Path, dpi: int) -> bool:
    if not any(has_key(log, "total_steps") or has_key(log, "ppo_updates") for log in logs.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plotted = False
    for name, log in logs.items():
        episodes = episodes_for(log, "rewards")
        if has_key(log, "total_steps"):
            steps = float_array(log, "total_steps")
            n = min(len(steps), len(episodes))
            axes[0].plot(episodes[:n], steps[:n], color=color_for(name), linewidth=1.8, label=label_for(name))
            plotted = True
        if has_key(log, "ppo_updates"):
            updates = float_array(log, "ppo_updates")
            n = min(len(updates), len(episodes))
            axes[1].step(episodes[:n], updates[:n], where="post", color=color_for(name), linewidth=1.8, label=label_for(name))
            plotted = True

    axes[0].set_title("Environment Steps", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Steps")
    axes[1].set_title("PPO Update Count", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Updates")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_current(out_path, dpi)
    return plotted


def plot_scene_training(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    dpi: int,
    max_steps: int,
) -> bool:
    scene_ids = sorted(
        {
            int(scene_id)
            for log in logs.values()
            if has_key(log, "scene_ids")
            for scene_id in get_array(log, "scene_ids")
        }
    )
    if not scene_ids:
        return False

    names = list(logs.keys())
    x = np.arange(len(scene_ids))
    width = min(0.8 / max(1, len(names)), 0.22)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    for ax, metric_name, ylabel, ylim in [
        (axes[0], "coverage", "Mean Coverage (%)", (0, 105)),
        (axes[1], "task_score", "Mean Task Score (%)", (0, 105)),
    ]:
        for i, name in enumerate(names):
            log = logs[name]
            scene_arr = get_array(log, "scene_ids").astype(int)
            if metric_name == "coverage":
                values = float_array(log, "coverages") * 100.0
            else:
                values = task_scores(log, max_steps)

            vals = []
            for scene_id in scene_ids:
                mask = scene_arr == scene_id
                vals.append(float(np.nanmean(values[mask])) if np.any(mask) else np.nan)

            offset = (i - (len(names) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.92, color=color_for(name), label=label_for(name), alpha=0.86)

        ax.set_title(f"Training {ylabel} by Scene", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Scene")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scene_ids])
        ax.set_ylim(*ylim)

    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_last_window_summary(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    out_path: Path,
    dpi: int,
    max_steps: int,
    last_n: int = 100,
) -> bool:
    names = list(logs.keys())
    labels = [label_for(name) for name in names]
    colors = [color_for(name) or "#777777" for name in names]

    metrics = [
        ("reward", "Mean Reward"),
        ("task_score", "Task Score (%)"),
        ("coverage", "Coverage (%)"),
        ("success", "Success Rate (%)"),
        ("steps", "Episode Steps"),
        ("timeout", "Timeout Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    plotted = False
    for ax, (metric_name, title) in zip(axes.ravel(), metrics):
        vals = []
        for name in names:
            log = logs[name]
            if metric_name == "reward" and has_key(log, "rewards"):
                arr = float_array(log, "rewards")
            elif metric_name == "task_score":
                arr = task_scores(log, max_steps)
            elif metric_name == "coverage" and has_key(log, "coverages"):
                arr = float_array(log, "coverages") * 100.0
            elif metric_name == "success" and has_key(log, "success"):
                arr = float_array(log, "success") * 100.0
            elif metric_name == "steps" and has_key(log, "lengths"):
                arr = float_array(log, "lengths")
            elif metric_name == "timeout" and has_key(log, "timeout"):
                arr = float_array(log, "timeout") * 100.0
            else:
                arr = np.asarray([], dtype=float)
            vals.append(float(np.nanmean(arr[-last_n:])) if arr.size else np.nan)

        if np.all(np.isnan(vals)):
            continue
        bars = ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.86)
        ax.set_title(f"Last {last_n}: {title}", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(title)
        if title.endswith("(%)"):
            ax.set_ylim(0, 105)
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        plotted = True

    if not plotted:
        plt.close()
        return False
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def metric_value(metrics: Mapping, path: List[str]):
    value = metrics
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def plot_quality_summary(metrics: Mapping[str, Dict], out_path: Path, dpi: int) -> bool:
    if not any(metrics.values()):
        return False

    names = [name for name, data in metrics.items() if data]
    if not names:
        return False
    labels = [label_for(name) for name in names]
    colors = [color_for(name) or "#777777" for name in names]

    specs = [
        (["convergence_efficiency", "auc_task_score_by_steps"], "AUC Task Score (%)", 100.0),
        (["convergence_efficiency", "steps_to_threshold"], "Steps to Threshold", 1.0),
        (["reward_stability", "task_score_std_tail"], "Tail Task Std (%)", 100.0),
        (["kl_stability", "kl_overshoot_rate"], "KL Overshoot Rate (%)", 100.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    plotted = False
    for ax, (path, title, scale) in zip(axes.ravel(), specs):
        vals = []
        for name in names:
            value = metric_value(metrics[name], path)
            vals.append(float(value) * scale if value is not None else np.nan)
        if np.all(np.isnan(vals)):
            continue
        bars = ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.86)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(title)
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2g}", ha="center", va="bottom", fontsize=8)
        plotted = True

    if not plotted:
        plt.close()
        return False
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def write_summary_csv(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Dict],
    out_path: Path,
    max_steps: int,
    last_n: int = 100,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "run",
            "episodes",
            "final_stage",
            "ppo_updates",
            "last_task_score_percent",
            "last_coverage_percent",
            "last_success_percent",
            "last_timeout_percent",
            "auc_task_score_percent",
            "steps_to_threshold",
            "tail_task_score_std_percent",
            "kl_overshoot_percent",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, log in logs.items():
            scores = task_scores(log, max_steps)
            coverage = float_array(log, "coverages") * 100.0 if has_key(log, "coverages") else np.asarray([])
            success = float_array(log, "success") * 100.0 if has_key(log, "success") else np.asarray([])
            timeout = float_array(log, "timeout") * 100.0 if has_key(log, "timeout") else np.asarray([])
            stages = float_array(log, "stage") if has_key(log, "stage") else np.asarray([])
            updates = float_array(log, "ppo_updates") if has_key(log, "ppo_updates") else np.asarray([])
            data = metrics.get(name, {})
            writer.writerow(
                {
                    "run": name,
                    "episodes": len(scores),
                    "final_stage": int(stages[-1]) if stages.size else "",
                    "ppo_updates": int(updates[-1]) if updates.size else "",
                    "last_task_score_percent": f"{np.nanmean(scores[-last_n:]):.6f}" if scores.size else "",
                    "last_coverage_percent": f"{np.nanmean(coverage[-last_n:]):.6f}" if coverage.size else "",
                    "last_success_percent": f"{np.nanmean(success[-last_n:]):.6f}" if success.size else "",
                    "last_timeout_percent": f"{np.nanmean(timeout[-last_n:]):.6f}" if timeout.size else "",
                    "auc_task_score_percent": _fmt_metric(metric_value(data, ["convergence_efficiency", "auc_task_score_by_steps"]), 100.0),
                    "steps_to_threshold": _fmt_metric(metric_value(data, ["convergence_efficiency", "steps_to_threshold"]), 1.0),
                    "tail_task_score_std_percent": _fmt_metric(metric_value(data, ["reward_stability", "task_score_std_tail"]), 100.0),
                    "kl_overshoot_percent": _fmt_metric(metric_value(data, ["kl_stability", "kl_overshoot_rate"]), 100.0),
                }
            )


def _fmt_metric(value, scale: float) -> str:
    return "" if value is None else f"{float(value) * scale:.6f}"


def clean_previous_outputs(out_dir: Path) -> None:
    for pattern in ["fig*.png", "training_summary.csv"]:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def generate_figures(
    logs: Mapping[str, Mapping[str, np.ndarray]],
    configs: Mapping[str, Dict],
    metrics: Mapping[str, Dict],
    out_dir: Path,
    window: int,
    dpi: int,
    max_steps: int,
    aggregate_seeds: bool = False,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(out_dir)
    created: List[Path] = []

    if aggregate_seeds:
        specs = [
            ("fig01_training_reward.png", lambda p: plot_aggregate_metric(logs, "rewards", "Episode Reward", "Training Reward (mean +/- seed std)", p, window, dpi, max_steps)),
            ("fig02_training_task_score.png", lambda p: plot_aggregate_metric(logs, "task_score", "Task Score (%)", "Training Task Score (mean +/- seed std)", p, window, dpi, max_steps, ylim=(0, 105))),
            ("fig03_boundary_coverage.png", lambda p: plot_aggregate_metric(logs, "coverage", "Boundary Coverage (%)", "Training Boundary Coverage (mean +/- seed std)", p, window, dpi, max_steps, ylim=(0, 105))),
            ("fig04_moving_success_rate.png", lambda p: plot_aggregate_metric(logs, "success", "Success Rate (%)", f"Moving Success Rate (Window={window}, mean +/- seed std)", p, window, dpi, max_steps, ylim=(0, 105))),
            ("fig05_episode_length.png", lambda p: plot_aggregate_metric(logs, "steps", "Episode Steps", "Episode Length (mean +/- seed std)", p, window, dpi, max_steps, ylim=(0, max_steps * 1.05))),
            ("fig08_timeout_rates.png", lambda p: plot_aggregate_two_panel(logs, [("timeout", "Rate (%)", "Timeout Rate"), ("zero_coverage_timeout", "Rate (%)", "Zero-coverage Timeout Rate")], p, window, dpi, max_steps, ylims=[(0, 105), (0, 105)])),
            ("fig10_actor_critic_loss.png", lambda p: plot_aggregate_two_panel(logs, [("actor_loss", "Actor Loss", "Actor Loss"), ("critic_loss", "Critic Loss", "Critic Loss")], p, window, dpi, max_steps)),
            ("fig11_entropy.png", lambda p: plot_aggregate_metric(logs, "entropy", "Policy Entropy", "Policy Entropy (mean +/- seed std)", p, window, dpi, max_steps)),
            ("fig14_training_progress_budget.png", lambda p: plot_aggregate_two_panel(logs, [("total_steps", "Total Steps", "Environment Steps"), ("ppo_updates", "Updates", "PPO Update Count")], p, window, dpi, max_steps)),
            ("fig15_last100_training_summary.png", lambda p: plot_aggregate_last_window_summary(logs, p, dpi, max_steps, last_n=100)),
            ("fig16_model_quality_summary.png", lambda p: plot_aggregate_quality_summary(metrics, p, dpi)),
        ]
    else:
        specs = [
        ("fig01_training_reward.png", lambda p: plot_smoothed_metric(logs, "rewards", "Episode Reward", "Training Reward", p, window, dpi)),
        ("fig02_training_task_score.png", lambda p: plot_task_score(logs, p, window, dpi, max_steps)),
        ("fig03_boundary_coverage.png", lambda p: plot_smoothed_metric(logs, "coverages", "Boundary Coverage (%)", "Training Boundary Coverage", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
        ("fig04_moving_success_rate.png", lambda p: plot_smoothed_metric(logs, "success", "Success Rate (%)", f"Moving Success Rate (Window={window})", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
        ("fig05_episode_length.png", lambda p: plot_smoothed_metric(logs, "lengths", "Episode Steps", "Episode Length", p, window, dpi, ylim=(0, max_steps * 1.05))),
        ("fig06_curriculum_stage.png", lambda p: plot_stage_curve(logs, p, dpi)),
        ("fig07_done_reason_distribution.png", lambda p: plot_done_reasons(logs, p, dpi, last_n=100)),
        ("fig08_timeout_rates.png", lambda p: plot_timeout_rates(logs, p, window, dpi)),
        ("fig09_training_by_scene.png", lambda p: plot_scene_training(logs, p, dpi, max_steps)),
        ("fig10_actor_critic_loss.png", lambda p: plot_loss_curves(logs, p, window, dpi)),
        ("fig11_entropy.png", lambda p: plot_smoothed_metric(logs, "entropy", "Policy Entropy", "Policy Entropy", p, window, dpi)),
        ("fig12_ppo_kl_clip.png", lambda p: plot_ppo_diagnostics(logs, configs, p, dpi)),
        ("fig13_actor_lr_actions.png", lambda p: plot_learning_rate(logs, p, dpi)),
        ("fig14_training_progress_budget.png", lambda p: plot_progress_budget(logs, p, dpi)),
        ("fig15_last100_training_summary.png", lambda p: plot_last_window_summary(logs, p, dpi, max_steps, last_n=100)),
        ("fig16_model_quality_summary.png", lambda p: plot_quality_summary(metrics, p, dpi)),
        ]

    for filename, maker in specs:
        out_path = out_dir / filename
        if maker(out_path) and out_path.exists():
            created.append(out_path)

    write_summary_csv(logs, metrics, out_dir / "training_summary.csv", max_steps=max_steps, last_n=100)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training figures from saved CTDE-PPO baseline logs.")
    parser.add_argument(
        "--comparison-dir",
        "--results-dir",
        dest="results_dir",
        default=None,
        help="Result directory, package directory, variant output directory, or old comparison directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output figure directory. Defaults to <run>/figures/training_figures.",
    )
    parser.add_argument("--window", type=int, default=100, help="Moving-average window for episode curves.")
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum episode steps for y-axis limits.")
    parser.add_argument("--run-filter", default=None, help="Only plot runs whose name contains this text, e.g. seed42.")
    parser.add_argument("--aggregate-seeds", action="store_true", help="Group *_seedN runs by method and plot mean +/- seed std.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_stdout()
    configure_matplotlib()

    outputs_dir = Path(__file__).resolve().parent
    result_dir = resolve_results_dir(args.results_dir, outputs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(result_dir)
    if not out_dir.is_absolute():
        out_dir = outputs_dir / out_dir

    logs, configs, metrics = load_logs(result_dir)
    logs, configs, metrics = filter_runs(logs, configs, metrics, args.run_filter)
    max_steps = max_steps_for(configs, args.max_steps)
    created = generate_figures(
        logs=logs,
        configs=configs,
        metrics=metrics,
        out_dir=out_dir,
        window=args.window,
        dpi=args.dpi,
        max_steps=max_steps,
        aggregate_seeds=args.aggregate_seeds,
    )

    print(f"Results directory: {result_dir}")
    print(f"Figure directory: {out_dir}")
    print(f"Loaded runs: {', '.join(logs.keys())}")
    print(f"Generated {len(created)} training figures:")
    for path in created:
        print(f"  - {path.name}")
    print("  - training_summary.csv")


if __name__ == "__main__":
    main()
