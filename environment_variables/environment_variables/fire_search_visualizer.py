"""Interactive viewer for a trained CTDE-PPO fire-search policy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.widgets import Button, RadioButtons

from ctde_ppo_baseline_train import (
    _make_eval_agent,
    _persistent_env_kwargs,
    normalize_training_config,
    set_seed,
)
from rl_environment_baseline import FireSearchBaselineEnvironment
from 信息转换 import DatasetIndex


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
SPLIT_ORDER = ("train", "validation", "generalization", "stress")
BEST_MODEL_ORDER = (
    "ppo_best_full_coverage.pth",
    "ppo_best_val.pth",
    "ppo_best.pth",
    "ppo_best_train.pth",
)
FINAL_MODEL_NAMES = {"ppo_final.pth", "ppo_post_target_final.pth"}


def _model_root(model_path: Path) -> Path:
    return model_path.parent.parent


def _config_path(model_path: Path) -> Path:
    return _model_root(model_path) / "config.json"


def find_model_files() -> List[Path]:
    if not OUTPUTS_DIR.is_dir():
        return []
    return sorted(
        OUTPUTS_DIR.rglob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_latest_completed_best(models: List[Path]) -> Optional[Path]:
    finals = [path for path in models if path.name in FINAL_MODEL_NAMES]
    finals.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for final_path in finals:
        model_dir = final_path.parent
        for name in BEST_MODEL_ORDER:
            candidate = model_dir / name
            if candidate.is_file() and _config_path(candidate).is_file():
                return candidate
        if _config_path(final_path).is_file():
            return final_path
    return None


def _choose(title: str, options: List[str]) -> Optional[int]:
    print(f"\n{title}")
    for index, label in enumerate(options, start=1):
        print(f"[{index}] {label}")
    print("[0] 返回")

    while True:
        try:
            value = input("请输入编号：").strip()
        except EOFError:
            return None
        if value == "0":
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return int(value) - 1
        print("输入无效，请输入列表中的编号。")


def choose_split(dataset_index: DatasetIndex) -> Optional[str]:
    splits = [name for name in SPLIT_ORDER if dataset_index.splits.get(name)]
    labels = [f"{name}（{len(dataset_index.splits[name])}个场景）" for name in splits]
    selected = _choose("请选择数据模式", labels)
    return None if selected is None else splits[selected]


def choose_scene(dataset_index: DatasetIndex, split: str) -> Optional[str]:
    scene_keys = dataset_index.scene_keys(split)
    labels = []
    for index, scene_key in enumerate(scene_keys, start=1):
        record = dataset_index.scenes[scene_key]
        scene_dir = record.get("scene_dir", "")
        difficulty = record.get("difficulty", "unknown")
        labels.append(
            f"场景{index} | {scene_key} | {scene_dir} | 难度={difficulty}"
        )
    selected = _choose(f"当前模式：{split}，请选择场景", labels)
    return None if selected is None else scene_keys[selected]


def choose_model(models: List[Path]) -> Optional[Path]:
    latest_best = find_latest_completed_best(models)
    latest_checkpoint = models[0] if models else None
    choices = []
    values: List[Optional[Path]] = []
    if latest_best is not None:
        choices.append(f"最新完整训练的最佳模型（推荐）| {latest_best}")
        values.append(latest_best)
    if latest_checkpoint is not None:
        choices.append(f"最新checkpoint | {latest_checkpoint}")
        values.append(latest_checkpoint)
    choices.append("从最近20个模型中手动选择")
    values.append(None)

    selected = _choose("请选择策略模型", choices)
    if selected is None:
        return None
    if values[selected] is not None:
        return values[selected]

    recent = models[:20]
    labels = [
        f"{datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S} | "
        f"{path.name} | {_model_root(path).name}"
        for path in recent
    ]
    manual = _choose("请选择具体模型", labels)
    return None if manual is None else recent[manual]


def load_config(model_path: Path) -> Dict:
    config_path = _config_path(model_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"模型缺少对应的config.json：{config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = normalize_training_config(json.load(file))
    config["data_dir"] = str(DATASET_DIR)
    return config


def build_runtime(
    split: str,
    scene_key: str,
    model_path: Path,
) -> Tuple[FireSearchBaselineEnvironment, object, Dict, Dict]:
    config = load_config(model_path)
    set_seed(int(config["seed"]))
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
        mode=split,
        fixed_scene_key=scene_key,
        init_area_percent=config["init_area_percent"],
        stage2_target=config["stage2_success_target"],
        stage3_target=config["stage3_success_target"],
        stage3_near_prob=config["stage3_near_prob"],
        termination_mode=config["evaluation_mode"],
        **_persistent_env_kwargs(config),
    )
    observation = env.reset()
    agent = _make_eval_agent(config, env)
    try:
        agent.load(str(model_path), restore_training_state=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "模型结构与当前环境配置不兼容。请检查模型的observation_profile、"
            "无人机数量和对应config.json。"
        ) from exc
    agent.actor.eval()
    agent.critic.eval()
    if agent.role_agent is not None:
        agent.role_agent.actor.eval()
        agent.role_agent.critic.eval()
    return env, agent, observation, config


def _mask_rgba(mask: np.ndarray, color: Tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = np.asarray(mask, dtype=np.float32) * alpha
    return rgba


class FireSearchViewer:
    def __init__(
        self,
        env: FireSearchBaselineEnvironment,
        agent: object,
        observation: Dict,
        config: Dict,
        split: str,
        scene_key: str,
        model_path: Path,
    ):
        self.env = env
        self.agent = agent
        self.observation = observation
        self.config = config
        self.split = split
        self.scene_key = scene_key
        self.model_path = model_path
        self.paused = True
        self.done = False
        self.done_reason = "等待开始"
        self.cumulative_reward = 0.0
        self.coverage = 0.0
        self.exact_coverage = 0.0
        self.tolerant_coverage = 0.0
        self.communication_rate = 0.0
        self.revisit_ratio = 0.0
        self.team_overlap_ratio = 0.0
        self.invalid_action_count = 0
        self.thermal_visible = True
        self.trails = [[position.copy()] for position in env.drone_positions]

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self.figure = plt.figure(figsize=(13.5, 8.2))
        grid = self.figure.add_gridspec(1, 2, width_ratios=(3.2, 1.35))
        self.map_ax = self.figure.add_subplot(grid[0, 0])
        self.info_ax = self.figure.add_subplot(grid[0, 1])
        self.figure.subplots_adjust(bottom=0.16, left=0.05, right=0.98, top=0.94)
        self._create_map_artists()
        self._create_controls()
        self._update_artists()

        self.timer = self.figure.canvas.new_timer(interval=250)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

    def _create_map_artists(self) -> None:
        terrain = self.env.env_data.data.get("elevation")
        if terrain is None:
            terrain = np.zeros(self.env.grid_size, dtype=np.float32)
        self.map_ax.imshow(terrain, cmap="terrain", origin="upper")
        thermal = getattr(self.env.env_data, "thermal_field", None)
        if thermal is None:
            thermal = np.zeros(self.env.grid_size, dtype=np.float32)
        self.thermal_image = self.map_ax.imshow(
            np.clip(np.asarray(thermal, dtype=np.float32), 0.0, 1.0),
            cmap="inferno",
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            alpha=0.38,
            interpolation="bilinear",
        )
        self.thermal_colorbar = self.figure.colorbar(
            self.thermal_image, ax=self.map_ax, fraction=0.035, pad=0.02
        )
        self.thermal_colorbar.set_label("归一化温度场")
        empty = np.zeros(self.env.grid_size, dtype=np.bool_)
        self.fire_image = self.map_ax.imshow(
            _mask_rgba(empty, (1.0, 0.1, 0.0), 0.48), origin="upper"
        )
        self.discovered_image = self.map_ax.imshow(
            _mask_rgba(empty, (0.0, 0.85, 1.0), 0.34), origin="upper"
        )
        self.true_boundary = self.map_ax.scatter([], [], s=5, c="#ff5b3a", alpha=0.8)
        self.found_boundary = self.map_ax.scatter(
            [], [], s=13, c="#ffe34d", edgecolors="black", linewidths=0.25
        )

        colors = ["#1665d8", "#8a2be2", "#00a878", "#ff8c00"]
        self.drone_points = []
        self.trail_lines = []
        self.sensor_circles = []
        for drone_index in range(self.env.num_drones):
            color = colors[drone_index % len(colors)]
            line, = self.map_ax.plot([], [], color=color, linewidth=1.4, alpha=0.9)
            point = self.map_ax.scatter(
                [], [], s=80, c=color, marker="^", edgecolors="white", linewidths=0.8
            )
            circle = Circle(
                (0, 0), self.env.vision_radius, fill=False, color=color,
                linestyle="--", linewidth=1.0, alpha=0.7
            )
            self.map_ax.add_patch(circle)
            self.trail_lines.append(line)
            self.drone_points.append(point)
            self.sensor_circles.append(circle)

        legend_items = [
            Patch(facecolor=plt.cm.inferno(0.75), alpha=0.38, label="归一化温度场"),
            Patch(facecolor=(1.0, 0.1, 0.0, 0.48), label="真实火场（仅绘图）"),
            Patch(facecolor=(0.0, 0.85, 1.0, 0.34), label="无人机已探测火区"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffe34d",
                   markeredgecolor="black", label="已发现边界"),
        ]
        self.map_ax.legend(handles=legend_items, loc="upper right", fontsize=9)
        self.map_ax.set_xlabel("列 / x")
        self.map_ax.set_ylabel("行 / y")

    def _create_controls(self) -> None:
        pause_ax = self.figure.add_axes([0.08, 0.045, 0.10, 0.055])
        step_ax = self.figure.add_axes([0.20, 0.045, 0.10, 0.055])
        restart_ax = self.figure.add_axes([0.32, 0.045, 0.10, 0.055])
        speed_ax = self.figure.add_axes([0.48, 0.018, 0.10, 0.12])
        thermal_ax = self.figure.add_axes([0.60, 0.045, 0.12, 0.055])
        self.pause_button = Button(pause_ax, "开始")
        self.step_button = Button(step_ax, "单步")
        self.restart_button = Button(restart_ax, "重新运行")
        self.thermal_button = Button(thermal_ax, "隐藏温度场")
        self.speed_buttons = RadioButtons(
            speed_ax, ("0.5x", "1x", "2x", "5x"), active=1
        )
        self.pause_button.on_clicked(self._toggle_pause)
        self.step_button.on_clicked(self._single_step)
        self.restart_button.on_clicked(self._restart)
        self.thermal_button.on_clicked(self._toggle_thermal)
        self.speed_buttons.on_clicked(self._set_speed)

    def _toggle_pause(self, _event) -> None:
        if self.done:
            return
        self.paused = not self.paused
        self.pause_button.label.set_text("继续" if self.paused else "暂停")

    def _single_step(self, _event) -> None:
        if not self.done:
            self._advance()

    def _restart(self, _event) -> None:
        set_seed(int(self.config["seed"]))
        self.observation = self.env.reset()
        self.done = False
        self.done_reason = "等待开始"
        self.paused = True
        self.pause_button.label.set_text("开始")
        self.cumulative_reward = 0.0
        self.coverage = 0.0
        self.exact_coverage = 0.0
        self.tolerant_coverage = 0.0
        self.communication_rate = 0.0
        self.revisit_ratio = 0.0
        self.team_overlap_ratio = 0.0
        self.invalid_action_count = 0
        self.trails = [[position.copy()] for position in self.env.drone_positions]
        self._update_artists()

    def _set_speed(self, label: str) -> None:
        speed = float(label.removesuffix("x"))
        self.timer.interval = max(25, int(250 / speed))

    def _toggle_thermal(self, _event) -> None:
        self.thermal_visible = not self.thermal_visible
        self.thermal_image.set_visible(self.thermal_visible)
        self.thermal_colorbar.ax.set_visible(self.thermal_visible)
        self.thermal_button.label.set_text(
            "隐藏温度场" if self.thermal_visible else "显示温度场"
        )
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        if not self.paused and not self.done:
            self._advance()

    def _advance(self) -> None:
        roles = self.agent.select_roles_if_required(
            self.observation, deterministic=True, track_option=False
        )
        if roles is not None:
            self.observation = self.env.apply_joint_role_assignment(roles)
        actions = self.agent.select_actions_deterministic(
            self.observation["local_obs"], self.observation["action_masks"]
        )
        self.observation, rewards, self.done, info = self.env.step(actions)
        self.cumulative_reward += float(np.mean(rewards))
        self.coverage = float(info.get("objective_coverage", info["boundary_coverage"]))
        self.exact_coverage = float(info["boundary_coverage"])
        self.tolerant_coverage = float(info.get("tolerant_boundary_coverage", 0.0))
        self.communication_rate = float(info.get("communication_available_rate", 0.0))
        self.revisit_ratio = float(info.get("pre_boundary_revisit_ratio", 0.0))
        self.team_overlap_ratio = float(info.get("team_overlap_ratio", 0.0))
        self.invalid_action_count = int(info.get("invalid_action_count", 0))
        self.done_reason = info["done_reason"]
        for trail, position in zip(self.trails, self.env.drone_positions):
            trail.append(position.copy())
        if self.done:
            self.paused = True
            self.pause_button.label.set_text("已结束")
        self._update_artists()

    def _update_artists(self) -> None:
        thermal = getattr(self.env.env_data, "thermal_field", None)
        if thermal is not None:
            self.thermal_image.set_data(
                np.clip(np.asarray(thermal, dtype=np.float32), 0.0, 1.0)
            )
        fire_mask = np.asarray(self.env.env_data.fire_binary_map) > 0
        self.fire_image.set_data(_mask_rgba(fire_mask, (1.0, 0.1, 0.0), 0.48))
        self.discovered_image.set_data(
            _mask_rgba(self.env.discovered_area_mask, (0.0, 0.85, 1.0), 0.34)
        )

        boundary = np.asarray(self.env.boundary_points, dtype=np.float32)
        boundary_xy = boundary[:, [1, 0]] if boundary.size else np.empty((0, 2))
        self.true_boundary.set_offsets(boundary_xy)
        found = np.asarray(sorted(self.env.discovered_boundary), dtype=np.float32)
        found_xy = found[:, [1, 0]] if found.size else np.empty((0, 2))
        self.found_boundary.set_offsets(found_xy)

        for index, (position, trail) in enumerate(zip(self.env.drone_positions, self.trails)):
            self.drone_points[index].set_offsets([[position[1], position[0]]])
            trail_array = np.asarray(trail)
            self.trail_lines[index].set_data(trail_array[:, 1], trail_array[:, 0])
            self.sensor_circles[index].center = (position[1], position[0])

        status = "已结束" if self.done else ("已暂停" if self.paused else "搜索中")
        drone_text = "\n".join(
            f"无人机{i + 1}: ({int(pos[0])}, {int(pos[1])})"
            for i, pos in enumerate(self.env.drone_positions)
        )
        role_names = ("SEARCH", "TRACK", "REACQUIRE")
        role_text = ""
        if self.env.task_coordinator is not None:
            role_text = "\n".join(
                f"无人机{i + 1}角色: {role_names[int(role)]}"
                for i, role in enumerate(self.env.task_coordinator.current_roles)
            )
        coverage_label = (
            "时效边界覆盖率"
            if self.config["reward_profile"] == "persistent_boundary"
            else "当前边界覆盖率"
        )
        self.info_ax.clear()
        self.info_ax.axis("off")
        self.info_ax.text(
            0.0,
            1.0,
            "无人机找火动态演示\n\n"
            f"模式：{self.split}\n"
            f"场景：{self.scene_key}\n"
            f"模型：{self.model_path.name}\n"
            f"训练变体：{self.config.get('variant_name', 'unknown')}\n"
            f"观测配置：{self.config['observation_profile']}\n"
            f"奖励配置：{self.config['reward_profile']}\n\n"
            f"状态：{status}\n"
            f"终止原因：{self.done_reason}\n"
            f"步数：{self.env.step_count} / {self.env.max_steps}\n"
            f"{coverage_label}：{self.coverage * 100:.2f}%\n"
            f"精确当前边界覆盖率：{self.exact_coverage * 100:.2f}%\n"
            f"容差覆盖率：{self.tolerant_coverage * 100:.2f}%\n"
            f"累计团队奖励：{self.cumulative_reward:.2f}\n"
            f"已发现边界点：{len(self.env.discovered_boundary)}\n"
            f"探测半径：{self.env.vision_radius}格\n\n"
            f"通信可用率：{self.communication_rate * 100:.1f}%\n"
            f"找火前重复率：{self.revisit_ratio * 100:.1f}%\n"
            f"团队重叠率：{self.team_overlap_ratio * 100:.1f}%\n"
            f"无效动作：{self.invalid_action_count}\n\n"
            f"{drone_text}\n{role_text}\n\n"
            "说明：红色真实火场和完整温度场\n只用于绘图，不会输入无人机策略。",
            ha="left",
            va="top",
            fontsize=10.5,
            linespacing=1.45,
        )
        self.map_ax.set_title(
            f"{self.split} / {self.scene_key} | step={self.env.step_count}"
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    dataset_index = DatasetIndex(str(DATASET_DIR))
    models = find_model_files()
    if not models:
        raise FileNotFoundError(f"没有在输出目录中找到.pth模型：{OUTPUTS_DIR}")

    while True:
        split = choose_split(dataset_index)
        if split is None:
            print("程序已退出。")
            return
        while True:
            scene_key = choose_scene(dataset_index, split)
            if scene_key is None:
                break
            model_path = choose_model(models)
            if model_path is None:
                continue
            print("\n正在加载场景和模型，请稍候……")
            try:
                env, agent, observation, config = build_runtime(
                    split, scene_key, model_path
                )
            except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                print(f"加载失败：{exc}")
                continue
            print(f"已加载模型：{model_path}")
            FireSearchViewer(
                env, agent, observation, config, split, scene_key, model_path
            ).show()


if __name__ == "__main__":
    main()
