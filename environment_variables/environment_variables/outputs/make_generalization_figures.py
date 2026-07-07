"""Generate generalization figures from saved CTDE-PPO evaluation records.

This script only plots data already saved by an evaluation run. It does not
train, reset environments, or run new evaluation episodes.

Supported inputs include:
    detailed_eval_*.csv
    _result_*.json
    eval_results*.json
    generalization_results*.json

The current baseline evaluate() function returns stage dictionaries containing
a records list. If those results are saved as JSON, this script can plot them.
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
BASELINE_NAME = "Baseline_CTDE_PPO"

RUN_ORDER = [
    BASELINE_NAME,
    "Fixed_LR_CTDE_PPO",
    "KL_LR_CTDE_PPO",
    "Full_SDF+Infotaxis",
    "No_SDF",
    "No_Infotaxis",
    "Vanilla_CTDE_PPO",
]

LABELS = {
    BASELINE_NAME: "Baseline CTDE-PPO",
    "Fixed_LR_CTDE_PPO": "Fixed LR CTDE-PPO",
    "KL_LR_CTDE_PPO": "KL-adaptive LR CTDE-PPO",
    "Full_SDF+Infotaxis": "Full (SDF + Infotaxis)",
    "No_SDF": "No SDF",
    "No_Infotaxis": "No Infotaxis",
    "Vanilla_CTDE_PPO": "Vanilla CTDE-PPO",
}

COLORS = {
    BASELINE_NAME: "#1f77b4",
    "Fixed_LR_CTDE_PPO": "#4c78a8",
    "KL_LR_CTDE_PPO": "#f58518",
    "Full_SDF+Infotaxis": "#d62728",
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


def to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_variant_name(name: str) -> str:
    name = str(name)
    if name in {"", RESULTS_DIR_NAME, "logs", "outputs"}:
        return BASELINE_NAME
    return name.replace("Full_SDF_Infotaxis", "Full_SDF+Infotaxis")


def base_run_name(name: str) -> str:
    text = str(name)
    marker = "_seed"
    if marker in text:
        prefix, suffix = text.rsplit(marker, 1)
        if suffix.isdigit():
            return prefix
    return text


def label_for(name: str) -> str:
    base = base_run_name(name)
    return LABELS.get(base, base)


def color_for(name: str):
    return COLORS.get(base_run_name(name), None)


def stage_from_key(key, default: int = 3) -> int:
    text = str(key)
    if text.startswith("stage_"):
        text = text.replace("stage_", "", 1)
    return to_int(text, default)


def _task_score(coverage: float, success: bool, steps: int, max_steps: int) -> float:
    efficiency = float(success) * (1.0 - np.clip(float(steps) / max(float(max_steps), 1.0), 0.0, 1.0))
    return float(0.5 * float(coverage) + 0.3 * float(success) + 0.2 * efficiency)


def infer_variant_from_path(path: Path) -> str:
    if path.name.startswith("_result_"):
        return normalize_variant_name(path.stem.replace("_result_", ""))
    if path.parent.name == "logs":
        return normalize_variant_name(path.parent.parent.name)
    return normalize_variant_name(path.parent.name)


def eval_files_in(path: Path) -> List[Path]:
    patterns = [
        "detailed_eval_*.csv",
        "*eval*.csv",
        "_result_*.json",
        "eval_results*.json",
        "*eval*_results*.json",
        "generalization_results*.json",
        "logs/eval_results*.json",
        "logs/*eval*_results*.json",
        "*/eval_results*.json",
        "*/logs/eval_results*.json",
        "*/logs/*eval*_results*.json",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))

    seen = set()
    unique = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen or not file_path.is_file():
            continue
        seen.add(resolved)
        unique.append(file_path)
    return sorted(unique)


def has_generalization_data(path: Path) -> bool:
    return path.is_dir() and bool(eval_files_in(path))


def result_dir_for_eval_file(path: Path) -> Path:
    if path.parent.name == "logs":
        output_dir = path.parent.parent
    else:
        output_dir = path.parent

    if output_dir.parent.name == RESULTS_DIR_NAME:
        return output_dir.parent
    return output_dir


def find_latest_results(outputs_dir: Path) -> Path:
    files = eval_files_in(outputs_dir)
    if not files:
        files = sorted(outputs_dir.glob("**/eval_results*.json"))
        files.extend(sorted(outputs_dir.glob("**/*eval*_results*.json")))
        files.extend(sorted(outputs_dir.glob("**/detailed_eval_*.csv")))
        files.extend(sorted(outputs_dir.glob("**/_result_*.json")))
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No saved generalization data found under {outputs_dir}")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return result_dir_for_eval_file(latest)


def resolve_results_dir(raw_dir: str | None, outputs_dir: Path) -> Path:
    if raw_dir is None:
        return find_latest_results(outputs_dir)

    candidate = Path(raw_dir)
    if not candidate.is_absolute():
        candidate = outputs_dir / candidate

    if (candidate / RESULTS_DIR_NAME).is_dir() and has_generalization_data(candidate / RESULTS_DIR_NAME):
        return candidate / RESULTS_DIR_NAME

    if has_generalization_data(candidate):
        return candidate

    files = eval_files_in(candidate)
    if files:
        return result_dir_for_eval_file(max(files, key=lambda p: p.stat().st_mtime))

    raise FileNotFoundError(f"No saved generalization data found in {candidate}")


def default_out_dir(result_dir: Path) -> Path:
    if result_dir.name == RESULTS_DIR_NAME:
        return result_dir.parent / "figures" / "generalization_figures"
    if result_dir.parent.name == RESULTS_DIR_NAME:
        return result_dir / "generalization_figures"
    return result_dir / "generalization_figures"


def normalize_record(raw: Mapping, variant: str, stage: int, index: int, max_steps: int) -> Dict:
    coverage = to_float(raw.get("coverage"), np.nan)
    if not np.isfinite(coverage) and raw.get("coverage_percent") is not None:
        coverage = to_float(raw.get("coverage_percent")) / 100.0
    coverage = 0.0 if not np.isfinite(coverage) else coverage

    steps = to_int(raw.get("steps", raw.get("length", raw.get("episode_length"))), 0)
    success = to_bool(raw.get("success", raw.get("mission_complete", False)))
    task_score = raw.get("task_score")
    if task_score is None or task_score == "":
        task_score = _task_score(coverage, success, steps, max_steps)
    else:
        task_score = to_float(task_score)
        if task_score > 1.5:
            task_score = task_score / 100.0

    done_reason = raw.get("done_reason") or ("mission_complete" if success else "other")
    timeout = raw.get("timeout")
    if timeout is None:
        timeout = done_reason == "max_steps_reached"

    return {
        "variant": normalize_variant_name(raw.get("variant_name") or raw.get("variant") or variant),
        "stage": to_int(raw.get("stage"), stage),
        "eval_episode": to_int(raw.get("eval_episode", raw.get("episode")), index),
        "scene_id": to_int(raw.get("scene_id"), -1),
        "seed": to_int(raw.get("seed"), 0),
        "reward": to_float(raw.get("reward")),
        "coverage": float(coverage),
        "success": bool(success),
        "steps": int(steps),
        "timeout": bool(to_bool(timeout)),
        "zero_coverage_timeout": bool(to_bool(raw.get("zero_coverage_timeout", False))),
        "task_score": float(task_score),
        "done_reason": str(done_reason),
        "info_gain": to_float(raw.get("info_gain", raw.get("information_gain")), 0.0),
    }


def read_csv_records(path: Path, max_steps: int) -> List[Dict]:
    records: List[Dict] = []
    variant = infer_variant_from_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            stage = to_int(row.get("stage"), stage_from_key(row.get("stage_key"), 3))
            records.append(normalize_record(row, variant, stage, idx, max_steps))
    return records


def iter_stage_items(data: Mapping) -> Iterable[Tuple[int, Mapping]]:
    if "eval_results" in data and isinstance(data["eval_results"], Mapping):
        data = data["eval_results"]
    elif "results" in data and isinstance(data["results"], Mapping):
        data = data["results"]

    if "records" in data or "episode_records" in data:
        yield to_int(data.get("stage"), 3), data
        return

    for key, value in data.items():
        if not isinstance(value, Mapping):
            continue
        if "records" in value or "episode_records" in value:
            yield stage_from_key(key, to_int(value.get("stage"), 3)), value


def read_json_records(path: Path, max_steps: int) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    variant = normalize_variant_name(data.get("variant_name") or data.get("variant") or infer_variant_from_path(path)) if isinstance(data, Mapping) else infer_variant_from_path(path)
    records: List[Dict] = []
    if isinstance(data, list):
        for idx, raw in enumerate(data, start=1):
            if isinstance(raw, Mapping):
                records.append(normalize_record(raw, variant, to_int(raw.get("stage"), 3), idx, max_steps))
        return records

    if not isinstance(data, Mapping):
        return records

    for stage, stage_result in iter_stage_items(data):
        raw_records = stage_result.get("records") or stage_result.get("episode_records") or []
        for idx, raw in enumerate(raw_records, start=len(records) + 1):
            if isinstance(raw, Mapping):
                records.append(normalize_record(raw, variant, stage, idx, max_steps))
    return records


def load_eval_records(result_dir: Path, max_steps: int) -> List[Dict]:
    records: List[Dict] = []
    for path in eval_files_in(result_dir):
        if path.suffix.lower() == ".csv":
            records.extend(read_csv_records(path, max_steps))
        elif path.suffix.lower() == ".json":
            records.extend(read_json_records(path, max_steps))

    if not records:
        raise RuntimeError(
            f"No episode-level generalization records found in {result_dir}. "
            "Save evaluate() output as eval_results.json, detailed_eval_*.csv, or _result_*.json first."
        )
    return records


def filter_records(records: List[Dict], run_filter: str | None) -> List[Dict]:
    if not run_filter:
        return records
    selected = [record for record in records if run_filter in str(record["variant"])]
    if not selected:
        raise RuntimeError(f"No generalization records matched filter: {run_filter}")
    return selected


def select_stage(records: List[Dict], requested_stage: int | None) -> Tuple[int, List[Dict]]:
    stages = sorted({int(record["stage"]) for record in records})
    stage = int(requested_stage) if requested_stage is not None else stages[-1]
    selected = [record for record in records if int(record["stage"]) == stage]
    if not selected:
        raise RuntimeError(f"No records found for stage {stage}; available stages: {stages}")
    return stage, selected


def ordered_variants(records: Iterable[Dict]) -> List[str]:
    seen = {record["variant"] for record in records}
    ordered = [name for name in RUN_ORDER if name in seen]
    ordered.extend(sorted(name for name in seen if name not in ordered))
    return ordered


def ordered_names(names: Iterable[str]) -> List[str]:
    seen = list(names)
    ordered = [name for name in RUN_ORDER if name in seen]
    ordered.extend(sorted(name for name in seen if name not in ordered))
    return ordered


def group_by_variant(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {name: [] for name in ordered_variants(records)}
    for record in records:
        grouped.setdefault(record["variant"], []).append(record)
    for values in grouped.values():
        values.sort(key=lambda item: (item["eval_episode"], item["scene_id"], item["seed"]))
    return grouped


def group_by_base_and_seed(records: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    for record in records:
        variant = str(record["variant"])
        grouped.setdefault(base_run_name(variant), {}).setdefault(variant, []).append(record)
    ordered = {name: grouped[name] for name in ordered_names(grouped)}
    for seed_groups in ordered.values():
        for values in seed_groups.values():
            values.sort(key=lambda item: (item["eval_episode"], item["scene_id"], item["seed"]))
    return ordered


def aligned_mean_std(series: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    series = [np.asarray(values, dtype=float) for values in series if np.asarray(values).size]
    if not series:
        return np.asarray([]), np.asarray([])
    n = min(len(values) for values in series)
    if n <= 0:
        return np.asarray([]), np.asarray([])
    arr = np.vstack([values[:n] for values in series])
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


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


def plot_smoothed_eval_metric(
    grouped: Mapping[str, List[Dict]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    window: int,
    dpi: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> bool:
    plt.figure(figsize=(9.0, 5.2))
    plotted = False

    for name, records in grouped.items():
        values = np.asarray([record[key] for record in records], dtype=float)
        if transform is not None:
            values = transform(values)
        x_smooth, y_smooth = rolling_mean(values, window)
        if x_smooth.size:
            plt.plot(x_smooth, y_smooth, color=color_for(name), linewidth=1.8, label=label_for(name))
            plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generalization Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_aggregate_smoothed_eval_metric(
    grouped: Mapping[str, Mapping[str, List[Dict]]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    window: int,
    dpi: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> bool:
    plt.figure(figsize=(9.0, 5.2))
    plotted = False
    for base, seed_groups in grouped.items():
        series = []
        for records in seed_groups.values():
            values = np.asarray([record[key] for record in records], dtype=float)
            if transform is not None:
                values = transform(values)
            _, y_smooth = rolling_mean(values, window)
            if y_smooth.size:
                series.append(y_smooth)
        mean, std = aligned_mean_std(series)
        if mean.size:
            x = np.arange(1, len(mean) + 1, dtype=float)
            color = color_for(base)
            plt.plot(x, mean, color=color, linewidth=2.0, label=label_for(base))
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generalization Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def scene_ids_for(records: Iterable[Dict]) -> List[int]:
    return sorted({int(record["scene_id"]) for record in records if int(record["scene_id"]) >= 0})


def mean_or_nan(values: List[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def plot_metric_by_scene(
    grouped: Mapping[str, List[Dict]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    transform: Callable[[float], float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> bool:
    scenes = scene_ids_for(record for records in grouped.values() for record in records)
    if not scenes:
        return False

    names = list(grouped.keys())
    x = np.arange(len(scenes))
    width = min(0.8 / max(1, len(names)), 0.22)

    plt.figure(figsize=(9.2, 5.2))
    for i, name in enumerate(names):
        vals = []
        for scene_id in scenes:
            raw_vals = [record[key] for record in grouped[name] if int(record["scene_id"]) == scene_id]
            if transform is not None:
                raw_vals = [transform(value) for value in raw_vals]
            vals.append(mean_or_nan(raw_vals))
        offset = (i - (len(names) - 1) / 2) * width
        plt.bar(x + offset, vals, width=width * 0.92, color=color_for(name), label=label_for(name), alpha=0.86)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generalization Scene", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x, [str(scene_id) for scene_id in scenes])
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_aggregate_metric_by_scene(
    grouped: Mapping[str, Mapping[str, List[Dict]]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    transform: Callable[[float], float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> bool:
    scenes = scene_ids_for(record for seed_groups in grouped.values() for records in seed_groups.values() for record in records)
    if not scenes:
        return False
    names = list(grouped.keys())
    x = np.arange(len(scenes))
    width = min(0.8 / max(1, len(names)), 0.22)
    plt.figure(figsize=(9.2, 5.2))
    for i, base in enumerate(names):
        means = []
        stds = []
        for scene_id in scenes:
            seed_vals = []
            for records in grouped[base].values():
                raw_vals = [record[key] for record in records if int(record["scene_id"]) == scene_id]
                if transform is not None:
                    raw_vals = [transform(value) for value in raw_vals]
                if raw_vals:
                    seed_vals.append(float(np.nanmean(raw_vals)))
            means.append(float(np.nanmean(seed_vals)) if seed_vals else np.nan)
            stds.append(float(np.nanstd(seed_vals)) if seed_vals else np.nan)
        offset = (i - (len(names) - 1) / 2) * width
        plt.bar(
            x + offset,
            means,
            yerr=stds,
            capsize=3,
            width=width * 0.92,
            color=color_for(base),
            label=label_for(base),
            alpha=0.86,
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generalization Scene", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x, [str(scene_id) for scene_id in scenes])
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_done_reasons(grouped: Mapping[str, List[Dict]], out_path: Path, dpi: int) -> bool:
    counters: Dict[str, Counter[str]] = {
        name: Counter(str(record["done_reason"]) for record in records)
        for name, records in grouped.items()
    }
    if not counters:
        return False

    preferred = ["mission_complete", "max_steps_reached", "battery_depleted", "other"]
    all_reasons = {reason for counter in counters.values() for reason in counter}
    categories = [reason for reason in preferred if reason in all_reasons]
    categories.extend(sorted(reason for reason in all_reasons if reason not in categories))

    names = list(grouped.keys())
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

    plt.title("Generalization Done Reason Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(x, [label_for(name) for name in names], rotation=18, ha="right")
    plt.ylim(0, 100)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_summary(grouped: Mapping[str, List[Dict]], out_path: Path, dpi: int) -> bool:
    names = list(grouped.keys())
    labels = [label_for(name) for name in names]
    colors = [color_for(name) or "#777777" for name in names]
    x = np.arange(len(names))
    metrics = [
        ("reward", "Mean Reward", lambda v: v),
        ("task_score", "Task Score (%)", lambda v: v * 100.0),
        ("coverage", "Coverage (%)", lambda v: v * 100.0),
        ("success", "Success Rate (%)", lambda v: v * 100.0),
        ("steps", "Episode Steps", lambda v: v),
        ("timeout", "Timeout Rate (%)", lambda v: v * 100.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    for ax, (key, title, transform) in zip(axes.ravel(), metrics):
        vals = []
        for name in names:
            raw_vals = [float(record[key]) for record in grouped[name]]
            vals.append(float(np.nanmean([transform(value) for value in raw_vals])))
        bars = ax.bar(x, vals, color=colors, alpha=0.86)
        ax.set_title(f"Generalization: {title}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(title)
        if title.endswith("(%)"):
            ax.set_ylim(0, 105)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_aggregate_summary(grouped: Mapping[str, Mapping[str, List[Dict]]], out_path: Path, dpi: int) -> bool:
    names = list(grouped.keys())
    labels = [label_for(name) for name in names]
    colors = [color_for(name) or "#777777" for name in names]
    x = np.arange(len(names))
    metrics = [
        ("reward", "Mean Reward", lambda v: v),
        ("task_score", "Task Score (%)", lambda v: v * 100.0),
        ("coverage", "Coverage (%)", lambda v: v * 100.0),
        ("success", "Success Rate (%)", lambda v: v * 100.0),
        ("steps", "Episode Steps", lambda v: v),
        ("timeout", "Timeout Rate (%)", lambda v: v * 100.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    for ax, (key, title, transform) in zip(axes.ravel(), metrics):
        means = []
        stds = []
        for base in names:
            seed_vals = []
            for records in grouped[base].values():
                raw_vals = [float(record[key]) for record in records]
                if raw_vals:
                    seed_vals.append(float(np.nanmean([transform(value) for value in raw_vals])))
            means.append(float(np.nanmean(seed_vals)) if seed_vals else np.nan)
            stds.append(float(np.nanstd(seed_vals)) if seed_vals else np.nan)
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.86)
        ax.set_title(f"Generalization: {title}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(title)
        if title.endswith("(%)"):
            ax.set_ylim(0, 105)
        for bar, val in zip(bars, means):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_score_distribution(grouped: Mapping[str, List[Dict]], out_path: Path, dpi: int) -> bool:
    names = list(grouped.keys())
    data = [[record["task_score"] * 100.0 for record in grouped[name]] for name in names]
    if not data:
        return False

    plt.figure(figsize=(9.2, 5.2))
    box = plt.boxplot(data, labels=[label_for(name) for name in names], patch_artist=True, showmeans=True)
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(color_for(name) or "#777777")
        patch.set_alpha(0.75)
    plt.title("Generalization Task Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Task Score (%)", fontsize=12)
    plt.xticks(rotation=18, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_efficiency_scatter(grouped: Mapping[str, List[Dict]], out_path: Path, dpi: int) -> bool:
    plt.figure(figsize=(9.2, 5.4))
    plotted = False
    for name, records in grouped.items():
        steps = np.asarray([record["steps"] for record in records], dtype=float)
        coverage = np.asarray([record["coverage"] * 100.0 for record in records], dtype=float)
        score = np.asarray([record["task_score"] * 100.0 for record in records], dtype=float)
        if steps.size == 0:
            continue
        plt.scatter(
            steps,
            coverage,
            s=np.clip(score, 10, 80),
            alpha=0.45,
            color=color_for(name),
            label=label_for(name),
            edgecolors="none",
        )
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title("Generalization Coverage vs Steps", fontsize=14, fontweight="bold")
    plt.xlabel("Episode Steps", fontsize=12)
    plt.ylabel("Boundary Coverage (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_current(out_path, dpi)
    return True


def plot_scene_heatmap(grouped: Mapping[str, List[Dict]], out_path: Path, dpi: int) -> bool:
    scenes = scene_ids_for(record for records in grouped.values() for record in records)
    names = list(grouped.keys())
    if not scenes or not names:
        return False

    matrix = np.full((len(names), len(scenes)), np.nan, dtype=float)
    for i, name in enumerate(names):
        for j, scene_id in enumerate(scenes):
            vals = [record["task_score"] * 100.0 for record in grouped[name] if int(record["scene_id"]) == scene_id]
            if vals:
                matrix[i, j] = float(np.nanmean(vals))

    fig, ax = plt.subplots(figsize=(max(8.0, len(scenes) * 0.75), max(3.8, len(names) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_title("Generalization Task Score Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scene")
    ax.set_ylabel("Run")
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels([str(scene) for scene in scenes])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([label_for(name) for name in names])
    for i in range(len(names)):
        for j in range(len(scenes)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value < 55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Task Score (%)")
    fig.tight_layout()
    save_current(out_path, dpi)
    return True


def write_summary_table(grouped: Mapping[str, List[Dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "run",
            "episodes",
            "mean_reward",
            "mean_task_score_percent",
            "mean_coverage_percent",
            "success_percent",
            "timeout_percent",
            "zero_coverage_timeout_percent",
            "mean_steps",
            "mean_info_gain",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, records in grouped.items():
            writer.writerow(
                {
                    "run": name,
                    "episodes": len(records),
                    "mean_reward": f"{np.nanmean([record['reward'] for record in records]):.6f}",
                    "mean_task_score_percent": f"{np.nanmean([record['task_score'] for record in records]) * 100.0:.6f}",
                    "mean_coverage_percent": f"{np.nanmean([record['coverage'] for record in records]) * 100.0:.6f}",
                    "success_percent": f"{np.nanmean([1.0 if record['success'] else 0.0 for record in records]) * 100.0:.6f}",
                    "timeout_percent": f"{np.nanmean([1.0 if record['timeout'] else 0.0 for record in records]) * 100.0:.6f}",
                    "zero_coverage_timeout_percent": f"{np.nanmean([1.0 if record['zero_coverage_timeout'] else 0.0 for record in records]) * 100.0:.6f}",
                    "mean_steps": f"{np.nanmean([record['steps'] for record in records]):.6f}",
                    "mean_info_gain": f"{np.nanmean([record['info_gain'] for record in records]):.6f}",
                }
            )


def clean_previous_outputs(out_dir: Path) -> None:
    for pattern in ["gen*.png", "generalization_summary.csv", "stage*_generalization_summary.csv"]:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def generate_figures(
    result_dir: Path,
    out_dir: Path,
    window: int,
    dpi: int,
    stage: int | None,
    max_steps: int,
    run_filter: str | None = None,
    aggregate_seeds: bool = False,
) -> List[Path]:
    records = filter_records(load_eval_records(result_dir, max_steps=max_steps), run_filter)
    selected_stage, selected_records = select_stage(records, stage)
    grouped = group_by_variant(selected_records)
    aggregate_grouped = group_by_base_and_seed(selected_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(out_dir)
    created: List[Path] = []
    if aggregate_seeds:
        specs = [
            ("gen01_generalization_task_score.png", lambda p: plot_aggregate_smoothed_eval_metric(aggregate_grouped, "task_score", "Task Score (%)", f"Stage {selected_stage} Generalization Task Score (mean +/- seed std)", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
            ("gen02_generalization_reward.png", lambda p: plot_aggregate_smoothed_eval_metric(aggregate_grouped, "reward", "Episode Reward", f"Stage {selected_stage} Generalization Reward (mean +/- seed std)", p, window, dpi)),
            ("gen03_generalization_coverage.png", lambda p: plot_aggregate_smoothed_eval_metric(aggregate_grouped, "coverage", "Boundary Coverage (%)", f"Stage {selected_stage} Generalization Boundary Coverage (mean +/- seed std)", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
            ("gen04_generalization_success_rate.png", lambda p: plot_aggregate_smoothed_eval_metric(aggregate_grouped, "success", "Success Rate (%)", f"Stage {selected_stage} Moving Generalization Success Rate (mean +/- seed std)", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
            ("gen05_generalization_episode_length.png", lambda p: plot_aggregate_smoothed_eval_metric(aggregate_grouped, "steps", "Episode Steps", f"Stage {selected_stage} Generalization Episode Length (mean +/- seed std)", p, window, dpi, ylim=(0, max_steps * 1.05))),
            ("gen07_generalization_task_score_by_scene.png", lambda p: plot_aggregate_metric_by_scene(aggregate_grouped, "task_score", "Task Score (%)", f"Stage {selected_stage} Task Score by Scene", p, dpi, transform=lambda value: value * 100.0, ylim=(0, 105))),
            ("gen08_generalization_coverage_by_scene.png", lambda p: plot_aggregate_metric_by_scene(aggregate_grouped, "coverage", "Mean Coverage (%)", f"Stage {selected_stage} Coverage by Scene", p, dpi, transform=lambda value: value * 100.0, ylim=(0, 105))),
            ("gen09_generalization_success_by_scene.png", lambda p: plot_aggregate_metric_by_scene(aggregate_grouped, "success", "Success Rate (%)", f"Stage {selected_stage} Success by Scene", p, dpi, transform=lambda value: 100.0 if value else 0.0, ylim=(0, 105))),
            ("gen10_generalization_timeout_by_scene.png", lambda p: plot_aggregate_metric_by_scene(aggregate_grouped, "timeout", "Timeout Rate (%)", f"Stage {selected_stage} Timeout by Scene", p, dpi, transform=lambda value: 100.0 if value else 0.0, ylim=(0, 105))),
            ("gen11_generalization_summary.png", lambda p: plot_aggregate_summary(aggregate_grouped, p, dpi)),
            ("gen12_task_score_distribution.png", lambda p: plot_score_distribution({base: [record for records in seed_groups.values() for record in records] for base, seed_groups in aggregate_grouped.items()}, p, dpi)),
            ("gen13_coverage_steps_efficiency.png", lambda p: plot_efficiency_scatter({base: [record for records in seed_groups.values() for record in records] for base, seed_groups in aggregate_grouped.items()}, p, dpi)),
            ("gen14_task_score_scene_heatmap.png", lambda p: plot_scene_heatmap({base: [record for records in seed_groups.values() for record in records] for base, seed_groups in aggregate_grouped.items()}, p, dpi)),
        ]
    else:
        specs = [
        ("gen01_generalization_task_score.png", lambda p: plot_smoothed_eval_metric(grouped, "task_score", "Task Score (%)", f"Stage {selected_stage} Generalization Task Score", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
        ("gen02_generalization_reward.png", lambda p: plot_smoothed_eval_metric(grouped, "reward", "Episode Reward", f"Stage {selected_stage} Generalization Reward", p, window, dpi)),
        ("gen03_generalization_coverage.png", lambda p: plot_smoothed_eval_metric(grouped, "coverage", "Boundary Coverage (%)", f"Stage {selected_stage} Generalization Boundary Coverage", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
        ("gen04_generalization_success_rate.png", lambda p: plot_smoothed_eval_metric(grouped, "success", "Success Rate (%)", f"Stage {selected_stage} Moving Generalization Success Rate", p, window, dpi, transform=lambda x: x * 100.0, ylim=(0, 105))),
        ("gen05_generalization_episode_length.png", lambda p: plot_smoothed_eval_metric(grouped, "steps", "Episode Steps", f"Stage {selected_stage} Generalization Episode Length", p, window, dpi, ylim=(0, max_steps * 1.05))),
        ("gen06_generalization_done_reason_distribution.png", lambda p: plot_done_reasons(grouped, p, dpi)),
        ("gen07_generalization_task_score_by_scene.png", lambda p: plot_metric_by_scene(grouped, "task_score", "Task Score (%)", f"Stage {selected_stage} Task Score by Scene", p, dpi, transform=lambda value: value * 100.0, ylim=(0, 105))),
        ("gen08_generalization_coverage_by_scene.png", lambda p: plot_metric_by_scene(grouped, "coverage", "Mean Coverage (%)", f"Stage {selected_stage} Coverage by Scene", p, dpi, transform=lambda value: value * 100.0, ylim=(0, 105))),
        ("gen09_generalization_success_by_scene.png", lambda p: plot_metric_by_scene(grouped, "success", "Success Rate (%)", f"Stage {selected_stage} Success by Scene", p, dpi, transform=lambda value: 100.0 if value else 0.0, ylim=(0, 105))),
        ("gen10_generalization_timeout_by_scene.png", lambda p: plot_metric_by_scene(grouped, "timeout", "Timeout Rate (%)", f"Stage {selected_stage} Timeout by Scene", p, dpi, transform=lambda value: 100.0 if value else 0.0, ylim=(0, 105))),
        ("gen11_generalization_summary.png", lambda p: plot_summary(grouped, p, dpi)),
        ("gen12_task_score_distribution.png", lambda p: plot_score_distribution(grouped, p, dpi)),
        ("gen13_coverage_steps_efficiency.png", lambda p: plot_efficiency_scatter(grouped, p, dpi)),
        ("gen14_task_score_scene_heatmap.png", lambda p: plot_scene_heatmap(grouped, p, dpi)),
        ]

    for filename, maker in specs:
        path = out_dir / filename
        if maker(path) and path.exists():
            created.append(path)

    write_summary_table(grouped, out_dir / "generalization_summary.csv")
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate generalization figures from saved evaluation records.")
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
        help="Output figure directory. Defaults to <run>/figures/generalization_figures.",
    )
    parser.add_argument("--window", type=int, default=10, help="Moving average window for evaluation episode curves.")
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    parser.add_argument("--stage", type=int, default=None, help="Evaluation stage to plot. Defaults to highest saved stage.")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum episode steps for score efficiency and y-axis.")
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

    created = generate_figures(
        result_dir=result_dir,
        out_dir=out_dir,
        window=args.window,
        dpi=args.dpi,
        stage=args.stage,
        max_steps=args.max_steps,
        run_filter=args.run_filter,
        aggregate_seeds=args.aggregate_seeds,
    )

    print(f"Results directory: {result_dir}")
    print(f"Figure directory: {out_dir}")
    print(f"Generated {len(created)} generalization figures:")
    for path in created:
        print(f"  - {path.name}")
    print("  - generalization_summary.csv")


if __name__ == "__main__":
    main()
