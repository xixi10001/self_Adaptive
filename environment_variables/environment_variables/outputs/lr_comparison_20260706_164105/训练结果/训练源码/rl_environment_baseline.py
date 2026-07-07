"""Clean baseline fire-search environment for CTDE-PPO.

It keeps the original task interface: decentralized local observations and a
centralized global state.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


_data_module = importlib.import_module("\u4fe1\u606f\u8f6c\u6362")
SceneManager = _data_module.SceneManager


class FireSearchBaselineEnvironment(gym.Env):
    """Baseline multi-drone fire boundary search environment."""

    OBSERVATION_PROFILE_DIMS = {
        "baseline": {"local_obs_dim": 17, "global_state_dim": 19},
        "static_terrain": {"local_obs_dim": 24, "global_state_dim": 19},
        "dynamic_front": {"local_obs_dim": 23, "global_state_dim": 19},
        "risk_aware": {"local_obs_dim": 20, "global_state_dim": 19},
    }
    REWARD_PROFILES = {
        "boundary_coverage",
        "front_detection",
        "severity_weighted",
        "exploration_balanced",
    }
    REWARD_BREAKDOWN_KEYS = [
        "r_discover",
        "r_coverage_gain",
        "r_area_gain",
        "r_boundary",
        "r_front",
        "r_severity",
        "r_explore",
        "r_penalty",
        "r_terminal",
    ]

    def __init__(
        self,
        data_dir: str = "./dataset",
        num_drones: int = 2,
        vision_radius: int = 16,
        max_steps: int = 600,
        use_metadata_uav_params: bool = False,
        observation_profile: str = "baseline",
        reward_profile: str = "boundary_coverage",
        curriculum_stage: int = 1,
        mode: str = "train",
        fixed_scene_key: Optional[str] = None,
        scene_keys: Optional[List[str]] = None,
        init_percentile: Optional[float] = 5.0,
        init_area_percent: Optional[float] = None,
        stage2_target: float = 0.15,
        stage3_target: float = 0.60,
        stage3_near_prob: float = 0.25,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_drones = int(num_drones)
        self.config_vision_radius = int(vision_radius)
        self.config_max_steps = int(max_steps)
        self.use_metadata_uav_params = bool(use_metadata_uav_params)
        self.vision_radius = self.config_vision_radius
        self.max_steps = self.config_max_steps
        self.observation_profile = self._validate_observation_profile(observation_profile)
        self.reward_profile = self._validate_reward_profile(reward_profile)
        self.curriculum_stage = int(curriculum_stage)
        self.mode = mode
        self.fixed_scene_key = fixed_scene_key
        self.scene_keys = [str(key) for key in scene_keys] if scene_keys is not None else None
        area_percent = init_area_percent if init_area_percent is not None else init_percentile
        self.init_area_percent = None if area_percent is None else float(area_percent)
        self.init_percentile = self.init_area_percent
        self.stage_targets = {2: float(stage2_target), 3: float(stage3_target)}
        self.stage3_near_prob = float(stage3_near_prob)

        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)

        self.coverage_gain_weight = 40.0
        self.coverage_gain_clip = 2.0
        self.stage1_explore_reward_cap = 25.0
        self.pre_boundary_area_gain_weight = 0.35
        self.pre_boundary_area_gain_clip = 0.08
        self.pre_boundary_repeat_window = 12
        self.pre_boundary_repeat_penalty = 0.04

        scene_keys_by_split = None
        if self.scene_keys is not None:
            split = SceneManager(data_dir).dataset_index.normalize_mode(mode)
            scene_keys_by_split = {split: self.scene_keys}
        self.scene_manager = SceneManager(data_dir, scene_keys_by_split=scene_keys_by_split)
        self._load_new_scene()

        dims = self.OBSERVATION_PROFILE_DIMS[self.observation_profile]
        self.local_obs_dim = dims["local_obs_dim"]
        self.global_state_dim = dims["global_state_dim"]
        self.observation_space = spaces.Dict(
            {
                "local_obs": spaces.Tuple(
                    tuple(
                        spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.local_obs_dim,),
                            dtype=np.float32,
                        )
                        for _ in range(self.num_drones)
                    )
                ),
                "global_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.global_state_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.max_battery = int(self.max_steps * 2.0)
        self.step_count = 0
        self.drone_positions: List[np.ndarray] = []
        self.drone_batteries: List[float] = []
        self.drone_momentums: List[np.ndarray] = []
        self.visited_cells = set()
        self.discovered_boundary = set()
        self.discovered_front = set()
        self.discovered_area_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.confirmed_boundary_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self._coverage_gradient = 0.0
        self._episode_explore_reward_total = 0.0
        self.first_heat_step = -1
        self.first_boundary_step = -1
        self._recent_cells: List[Tuple[int, int]] = []

        self.episode_reward_breakdown = self._empty_reward_breakdown()

        print(
            "基线环境已初始化 | "
            f"模式={mode} | 本地观测维度={self.local_obs_dim} | "
            f"全局状态维度={self.global_state_dim} | "
            f"observation_profile={self.observation_profile} | "
            f"reward_profile={self.reward_profile}"
        )

    def _load_new_scene(self):
        if self.fixed_scene_key is None:
            self.env_data = self.scene_manager.get_scene(self.mode)
        else:
            self.env_data = self.scene_manager.get_specific_scene(str(self.fixed_scene_key))

        boundary_points_t0 = self.env_data.initialize_training_boundary(
            init_percentile=self.init_percentile,
            init_area_percent=self.init_area_percent,
        )
        self.env_data.boundary_points = boundary_points_t0
        self.env_data._compute_thermal_field()

        self.grid_size = self.env_data.shape
        self._severity_map_cache = None
        self.boundary_points = list(self.env_data.boundary_points or [])
        self.total_boundary_points = max(len(self.boundary_points), 1)
        self.scene_id = self.env_data.scene_id
        self.scene_key = self.env_data.scene_key

        if self.boundary_points:
            self.fire_centroid = np.mean(np.array(self.boundary_points, dtype=np.float32), axis=0)
        else:
            self.fire_centroid = np.array(
                [self.grid_size[0] / 2.0, self.grid_size[1] / 2.0],
                dtype=np.float32,
            )

        self._build_boundary_set()
        self._apply_uav_params()
        # print(
        #     "Scene loaded | "
        #     f"scene_key={self.scene_key} | shape={self.grid_size} | "
        #     f"sensor_radius_cells={self.env_data.sensor_radius_cells} | max_steps={self.max_steps}"
        # )

    def _build_boundary_set(self):
        self._boundary_set = {(int(bp[0]), int(bp[1])) for bp in self.boundary_points}

    def _apply_uav_params(self):
        if self.use_metadata_uav_params:
            self.vision_radius = max(1, int(self.env_data.sensor_radius_cells))
            self.max_steps = max(1, int(self.env_data.max_steps))
        else:
            self.vision_radius = self.config_vision_radius
            self.max_steps = self.config_max_steps
        if hasattr(self, "max_battery"):
            self.max_battery = int(self.max_steps * 2.0)

    @classmethod
    def _validate_observation_profile(cls, profile: str) -> str:
        profile = str(profile).lower()
        if profile not in cls.OBSERVATION_PROFILE_DIMS:
            raise ValueError(
                f"Unknown observation_profile {profile!r}. "
                f"Expected one of: {sorted(cls.OBSERVATION_PROFILE_DIMS)}"
            )
        return profile

    @classmethod
    def _validate_reward_profile(cls, profile: str) -> str:
        profile = str(profile).lower()
        if profile not in cls.REWARD_PROFILES:
            raise ValueError(
                f"Unknown reward_profile {profile!r}. "
                f"Expected one of: {sorted(cls.REWARD_PROFILES)}"
            )
        return profile

    def _empty_reward_breakdown(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.REWARD_BREAKDOWN_KEYS}

    def _discovered_on_current_boundary_count(self) -> int:
        return sum(1 for p in self.discovered_boundary if p in self._boundary_set)

    def _boundary_coverage_rate(self) -> float:
        return self._discovered_on_current_boundary_count() / max(self.total_boundary_points, 1)

    def _get_circular_window(self, y: int, x: int):
        y_min = max(0, y - self.vision_radius)
        y_max = min(self.grid_size[0], y + self.vision_radius + 1)
        x_min = max(0, x - self.vision_radius)
        x_max = min(self.grid_size[1], x + self.vision_radius + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        local_mask = (yy - y) ** 2 + (xx - x) ** 2 <= self.vision_radius**2
        return y_min, y_max, x_min, x_max, local_mask

    def _mark_visible_region(self, pos: np.ndarray) -> int:
        y, x = int(pos[0]), int(pos[1])
        y_min, y_max, x_min, x_max, local_mask = self._get_circular_window(y, x)
        patch = self.discovered_area_mask[y_min:y_max, x_min:x_max]
        newly_visible = int(np.count_nonzero(~patch[local_mask]))
        patch[local_mask] = True
        return newly_visible

    def _new_visible_area_severity_stats(self, pos: np.ndarray) -> Tuple[int, float, float]:
        y, x = int(pos[0]), int(pos[1])
        y_min, y_max, x_min, x_max, local_mask = self._get_circular_window(y, x)
        patch = self.discovered_area_mask[y_min:y_max, x_min:x_max]
        new_mask = local_mask & ~patch
        new_count = int(np.count_nonzero(new_mask))
        if new_count == 0:
            return 0, 0.0, 0.0
        severity_patch = self._severity_map()[y_min:y_max, x_min:x_max]
        values = severity_patch[new_mask]
        return new_count, float(np.mean(values)), float(np.max(values))

    def _update_discovered_front(self, pos: np.ndarray) -> Tuple[int, int]:
        front_map = self._binary_front_map()
        total_front = int(np.count_nonzero(front_map))
        if total_front == 0:
            return 0, 0
        y, x = int(pos[0]), int(pos[1])
        y_min, y_max, x_min, x_max, local_mask = self._get_circular_window(y, x)
        local_front = (front_map[y_min:y_max, x_min:x_max] > 0) & local_mask
        points = np.argwhere(local_front)
        new_points = 0
        for point in points:
            cell = (int(point[0]) + y_min, int(point[1]) + x_min)
            if cell not in self.discovered_front:
                new_points += 1
            self.discovered_front.add(cell)
        return new_points, total_front

    def _get_visible_boundary_points(self, pos: np.ndarray) -> List[Tuple[int, int]]:
        if not self.boundary_points:
            return []
        y, x = int(pos[0]), int(pos[1])
        radius_sq = self.vision_radius**2
        visible = []
        for bp in self.boundary_points:
            by, bx = int(bp[0]), int(bp[1])
            dy = by - y
            dx = bx - x
            if dy * dy + dx * dx <= radius_sq:
                visible.append((by, bx))
        return visible

    def _refresh_confirmed_boundary_state(self):
        if not self.boundary_points:
            self.discovered_boundary = set()
            return
        refreshed = {
            (int(bp[0]), int(bp[1]))
            for bp in self.boundary_points
            if self.confirmed_boundary_mask[int(bp[0]), int(bp[1])]
        }
        self.discovered_boundary.update(refreshed)

    def reset(self) -> Dict:
        self._load_new_scene()

        self.step_count = 0
        self.visited_cells = set()
        self.discovered_boundary = set()
        self.discovered_front = set()
        self.discovered_area_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.confirmed_boundary_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self._coverage_gradient = 0.0
        self._episode_explore_reward_total = 0.0
        self.first_heat_step = -1
        self.first_boundary_step = -1
        self._recent_cells = []

        for key in self.episode_reward_breakdown:
            self.episode_reward_breakdown[key] = 0.0

        self.drone_positions = []
        self.drone_batteries = []
        self.drone_momentums = []
        self.spawn_modes = []

        for drone_idx in range(self.num_drones):
            pos = self._spawn_randomly(drone_idx)
            self.drone_positions.append(pos)
            self.drone_batteries.append(float(self.max_battery))
            self.drone_momentums.append(np.array([0.0, 0.0], dtype=np.float32))

        return self._get_observation()

    def _spawn_randomly(self, drone_idx: int) -> np.ndarray:
        if self._should_use_near_spawn():
            pos = self._spawn_near_boundary(drone_idx)
            if pos is not None:
                self.spawn_modes.append("near")
                return pos

        pos = self._spawn_far_from_fire()
        self.spawn_modes.append("far")
        return pos

    def _should_use_near_spawn(self) -> bool:
        if self.mode != "train":
            return False
        near_probs = {1: 0.70, 2: 0.50, 3: self.stage3_near_prob}
        return np.random.random() < near_probs.get(self.curriculum_stage, 0.0)

    def _spawn_near_boundary(self, drone_idx: int) -> Optional[np.ndarray]:
        if not self.boundary_points:
            return None

        h, w = self.grid_size
        margin = max(2, self.vision_radius // 2)
        
        if self.curriculum_stage == 1:
            min_dist = 0.0
            max_dist = float(self.vision_radius * 0.5)
        elif self.curriculum_stage == 2:
            min_dist = float(self.vision_radius * 0.5)
            max_dist = float(self.vision_radius * 1.5)
        else:
            min_dist = float(self.vision_radius)
            max_dist = float(self.vision_radius * 2.5)

        for _ in range(50):
            bp = np.array(
                self.boundary_points[np.random.randint(0, len(self.boundary_points))],
                dtype=np.float32,
            )
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            radius = np.random.uniform(min_dist, max_dist)
            offset = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * radius
            pos = np.rint(bp + offset).astype(np.float32)

            if not (margin <= pos[0] < h - margin and margin <= pos[1] < w - margin):
                continue
            dist_to_boundary = float(np.linalg.norm(pos - bp))
            if not (min_dist <= dist_to_boundary <= max_dist):
                continue
            if self._too_close_to_existing_drones(pos):
                continue
            return pos

        return None

    def _too_close_to_existing_drones(self, pos: np.ndarray) -> bool:
        min_spacing = float(self.vision_radius * 0.8)
        return any(np.linalg.norm(pos - other_pos) < min_spacing for other_pos in self.drone_positions)

    def _spawn_far_from_fire(self) -> np.ndarray:
        h, w = self.grid_size
        margin = max(2, self.vision_radius // 2)
        pos = np.array([h // 2, w // 2], dtype=np.float32)
        for _ in range(50):
            pos = np.array(
                [
                    np.random.randint(margin, h - margin),
                    np.random.randint(margin, w - margin),
                ],
                dtype=np.float32,
            )
            dist_to_fire = np.linalg.norm(pos - self.fire_centroid)
            if dist_to_fire > self.vision_radius * 2.5:
                return pos
        return pos

    def _base_cell_features(self, y: int, x: int) -> Dict[str, float]:
        data = self.env_data.data
        norm = self.env_data.norm_params
        features = {
            "intensity_norm": 0.0,
            "dem_norm": 0.0,
            "slope_norm": 0.0,
            "wind_speed_norm": 0.0,
            "wind_dir_sin": 0.0,
            "wind_dir_cos": 0.0,
        }

        if "intensity" in data:
            intensity = float(data["intensity"][y, x])
            if getattr(self.env_data, "fire_binary_map", None) is not None:
                if self.env_data.fire_binary_map[y, x] == 0:
                    intensity = 0.0
            features["intensity_norm"] = float(
                np.clip(intensity / max(float(norm["intensity_max"]), 1.0), 0.0, 1.0)
            )

        if "dem" in data:
            dem = float(data["dem"][y, x])
            denom = max(float(norm["dem_max"] - norm["dem_min"]), 1.0)
            features["dem_norm"] = float(
                np.clip((dem - float(norm["dem_min"])) / denom, 0.0, 1.0)
            )

        if "slope" in data:
            features["slope_norm"] = float(
                np.clip(
                    float(data["slope"][y, x]) / max(float(norm["slope_max"]), 1.0),
                    0.0,
                    1.0,
                )
            )

        if "wind_speed" in data:
            features["wind_speed_norm"] = float(
                np.clip(
                    float(data["wind_speed"][y, x])
                    / max(float(norm["wind_speed_max"]), 1.0),
                    0.0,
                    1.0,
                )
            )

        if "wind_direction" in data:
            wind_dir_rad = np.radians(float(data["wind_direction"][y, x]))
            features["wind_dir_sin"] = float(np.sin(wind_dir_rad))
            features["wind_dir_cos"] = float(np.cos(wind_dir_rad))

        return features

    def _normalized_static_value(self, key: str, y: int, x: int) -> float:
        if key not in self.env_data.data:
            return 0.0
        data = np.asarray(self.env_data.data[key], dtype=np.float32)
        max_value = max(float(np.nanmax(data)), 1.0)
        return float(np.clip(float(data[y, x]) / max_value, 0.0, 1.0))

    def _local_values(self, data: np.ndarray, y: int, x: int) -> np.ndarray:
        y_min, y_max, x_min, x_max, local_mask = self._get_circular_window(y, x)
        values = np.asarray(data[y_min:y_max, x_min:x_max], dtype=np.float32)[local_mask]
        return values[np.isfinite(values)]

    def _binary_front_map(self) -> np.ndarray:
        fire_map = getattr(self.env_data, "fire_binary_map", None)
        if fire_map is None:
            return np.zeros(self.grid_size, dtype=np.uint8)
        fire = np.asarray(fire_map > 0, dtype=np.bool_)
        padded = np.pad(fire, 1, mode="constant", constant_values=False)
        eroded = fire.copy()
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + fire.shape[0], dx : dx + fire.shape[1]]
        return (fire & ~eroded).astype(np.uint8)

    def _severity_map(self) -> np.ndarray:
        if self._severity_map_cache is None:
            self._severity_map_cache = self.env_data.severity_map()
        return self._severity_map_cache

    def _static_terrain_features(self, y: int, x: int) -> List[float]:
        aspect = float(self.env_data.data.get("aspect", np.zeros(self.grid_size))[y, x])
        aspect_rad = np.radians(aspect)
        return [
            float(np.sin(aspect_rad)),
            float(np.cos(aspect_rad)),
            self._normalized_static_value("fuel_model", y, x),
            self._normalized_static_value("canopy_cover", y, x),
            self._normalized_static_value("canopy_height", y, x),
            self._normalized_static_value("canopy_base_height", y, x),
            self._normalized_static_value("canopy_bulk_density", y, x),
        ]

    def _dynamic_front_features(
        self, y: int, x: int, local_fire_info: Dict[str, float], local_area: float
    ) -> List[float]:
        fire_map = getattr(self.env_data, "fire_binary_map", np.zeros(self.grid_size))
        front_map = self._binary_front_map()
        fire_values = self._local_values(fire_map, y, x)
        front_values = self._local_values(front_map, y, x)
        intensity_max = max(float(self.env_data.norm_params.get("intensity_max", 1.0)), 1.0)
        nearest = float(local_fire_info.get("nearest_fire_distance", float("inf")))
        if not np.isfinite(nearest):
            nearest = float(self.vision_radius)
        return [
            float(np.clip(np.sum(fire_values > 0) / local_area, 0.0, 1.0)),
            float(np.clip(np.sum(front_values > 0) / local_area, 0.0, 1.0)),
            float(np.clip(float(local_fire_info.get("boundary_count", 0)) / local_area, 0.0, 1.0)),
            float(np.clip(float(local_fire_info.get("avg_intensity", 0.0)) / intensity_max, 0.0, 1.0)),
            float(np.clip(float(local_fire_info.get("max_intensity", 0.0)) / intensity_max, 0.0, 1.0)),
            float(np.clip(nearest / max(float(self.vision_radius), 1.0), 0.0, 1.0)),
        ]

    def _risk_aware_features(self, y: int, x: int) -> List[float]:
        severity = self._severity_map()
        values = self._local_values(severity, y, x)
        if values.size == 0:
            return [0.0, 0.0, 0.0]
        return [
            float(np.clip(severity[y, x], 0.0, 1.0)),
            float(np.clip(np.mean(values), 0.0, 1.0)),
            float(np.clip(np.max(values), 0.0, 1.0)),
        ]

    def _get_observation(self) -> Dict:
        local_obs_list = []
        current_coverage_rate = self._boundary_coverage_rate()
        map_norm = max(float(np.linalg.norm(self.grid_size)), 1.0)
        local_area = max((self.vision_radius * 2 + 1) ** 2, 1)

        for i in range(self.num_drones):
            pos = self.drone_positions[i]
            battery = self.drone_batteries[i]
            y, x = int(pos[0]), int(pos[1])

            local_fire_info = self.env_data.get_local_fire_info(y, x, self.vision_radius)
            features = self._base_cell_features(y, x)
            map_center = np.array([self.grid_size[0] / 2.0, self.grid_size[1] / 2.0])
            dist_to_center = float(np.linalg.norm(pos - map_center) / map_norm)
            grad_y, grad_x = self.env_data.get_local_thermal_gradient(y, x)
            mom_y, mom_x = self.drone_momentums[i]
            cam_dir_y, cam_dir_x = local_fire_info["fire_direction"]

            local_obs = [
                pos[0] / self.grid_size[0],
                pos[1] / self.grid_size[1],
                battery / self.max_battery,
                features["intensity_norm"],
                float(local_fire_info["fire_count"]) / local_area,
                dist_to_center,
                features["wind_speed_norm"],
                features["wind_dir_sin"],
                features["wind_dir_cos"],
                features["dem_norm"],
                features["slope_norm"],
                float(grad_y),
                float(grad_x),
                float(mom_y),
                float(mom_x),
                float(cam_dir_y) / max(float(self.vision_radius), 1.0),
                float(cam_dir_x) / max(float(self.vision_radius), 1.0),
            ]
            if self.observation_profile == "static_terrain":
                local_obs.extend(self._static_terrain_features(y, x))
            elif self.observation_profile == "dynamic_front":
                local_obs.extend(
                    self._dynamic_front_features(y, x, local_fire_info, local_area)
                )
            elif self.observation_profile == "risk_aware":
                local_obs.extend(self._risk_aware_features(y, x))
            local_obs_list.append(np.array(local_obs, dtype=np.float32))

        avg_battery = float(np.mean(self.drone_batteries) / self.max_battery)
        min_battery = float(np.min(self.drone_batteries) / self.max_battery)
        team_centroid = np.mean(self.drone_positions, axis=0)
        team_spread = np.std(self.drone_positions, axis=0)
        dists_to_fire = [np.linalg.norm(pos - self.fire_centroid) for pos in self.drone_positions]
        avg_dist_to_fire = float(np.mean(dists_to_fire) / map_norm)

        wind_speeds = []
        elevations = []
        for pos in self.drone_positions:
            py, px = int(pos[0]), int(pos[1])
            features = self._base_cell_features(py, px)
            wind_speeds.append(features["wind_speed_norm"])
            elevations.append(features["dem_norm"])

        discovered_boundary_feature = (
            float(self._discovered_on_current_boundary_count()) / self.total_boundary_points
        )
        undiscovered_density = 1.0 - current_coverage_rate

        global_state = [
            current_coverage_rate,
            avg_battery,
            min_battery,
            team_centroid[0] / self.grid_size[0],
            team_centroid[1] / self.grid_size[1],
            team_spread[0] / self.grid_size[0],
            team_spread[1] / self.grid_size[1],
            avg_dist_to_fire,
            self.step_count / self.max_steps,
            len(self.visited_cells) / (self.grid_size[0] * self.grid_size[1]),
            float(self.curriculum_stage) / 3.0,
            float(np.mean(wind_speeds)) if wind_speeds else 0.0,
            float(np.mean(elevations)) if elevations else 0.0,
            discovered_boundary_feature,
            float(any(b < self.max_battery * 0.2 for b in self.drone_batteries)),
            float(self.num_drones),
            0.0,
            float(self._coverage_gradient),
            undiscovered_density,
        ]

        return {
            "local_obs": local_obs_list,
            "global_state": np.array(global_state, dtype=np.float32),
        }

    def _execute_action(self, pos: np.ndarray, action: int) -> np.ndarray:
        action_map = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
            4: np.array([0, 0]),
        }
        new_pos = pos + action_map[int(action)]
        return np.clip(new_pos, [0, 0], [self.grid_size[0] - 1, self.grid_size[1] - 1])

    def _get_heat_signal_features(self, pos: np.ndarray) -> Dict[str, float]:
        y, x = int(pos[0]), int(pos[1])
        curr_heat = max(0.0, self.env_data.get_thermal_value(y, x))
        local_fire_info = self.env_data.get_local_fire_info(y, x, self.vision_radius)
        intensity_max = max(float(self.env_data.norm_params.get("intensity_max", 1.0)), 1.0)
        local_peak_norm = float(np.clip(local_fire_info.get("max_intensity", 0.0) / intensity_max, 0.0, 1.0))
        has_heat_signal = (
            curr_heat >= 0.8
            or local_fire_info.get("fire_count", 0) > 0
            or local_peak_norm >= 0.02
        )
        return {"current_heat": float(curr_heat), "has_heat_signal": bool(has_heat_signal)}

    def _compute_reward(
        self,
        drone_id: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        action: int,
        peer_new_positions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        r_breakdown = self._empty_reward_breakdown()

        y, x = int(new_pos[0]), int(new_pos[1])
        cell = (y, x)
        is_boundary = cell in self._boundary_set

        if is_boundary and cell not in self.discovered_boundary:
            if self.curriculum_stage == 1:
                r_disc = 5.0
            elif self.curriculum_stage == 2:
                r_disc = 3.0
            else:
                r_disc = 2.0
            reward += r_disc
            r_breakdown["r_discover"] += r_disc
            r_breakdown["r_boundary"] += r_disc

        step_penalty = -0.02 if self.curriculum_stage == 1 else -0.08
        reward += step_penalty
        r_breakdown["r_penalty"] += step_penalty

        recent_window = self._recent_cells[-self.pre_boundary_repeat_window :]
        if len(self.discovered_boundary) == 0 and cell in recent_window:
            reward -= self.pre_boundary_repeat_penalty
            r_breakdown["r_penalty"] -= self.pre_boundary_repeat_penalty

        if cell not in self.visited_cells:
            explore_reward = 0.02
            if self.curriculum_stage == 1:
                remaining = max(0.0, self.stage1_explore_reward_cap - self._episode_explore_reward_total)
                explore_reward = min(explore_reward, remaining)
            if explore_reward > 0.0:
                reward += explore_reward
                r_breakdown["r_explore"] += explore_reward
                self._episode_explore_reward_total += explore_reward

        if int(action) == 4:
            idle_penalty = -0.10 if self.curriculum_stage == 1 else -0.25
            reward += idle_penalty
            r_breakdown["r_penalty"] += idle_penalty

        if is_boundary and cell in self.discovered_boundary:
            reward -= 0.10
            r_breakdown["r_penalty"] -= 0.10

        peers = peer_new_positions if peer_new_positions is not None else self.drone_positions
        if len(peers) > 1:
            for j, other_pos in enumerate(peers):
                if j == drone_id:
                    continue
                if np.linalg.norm(new_pos - other_pos) < self.vision_radius * 0.8:
                    reward -= 0.15
                    r_breakdown["r_penalty"] -= 0.15
                    break

        return float(reward), r_breakdown

    def _compute_profile_reward(
        self,
        pos: np.ndarray,
        cell_was_visited: bool,
        new_area_cells: int,
        severity_mean: float,
        severity_max: float,
    ) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        r_breakdown = self._empty_reward_breakdown()

        if self.reward_profile == "front_detection":
            new_front, total_front = self._update_discovered_front(pos)
            if new_front > 0:
                r_front = 20.0 * float(new_front) / max(float(total_front), 1.0)
                r_front = float(np.clip(r_front, 0.0, 1.0))
                reward += r_front
                r_breakdown["r_front"] += r_front

        elif self.reward_profile == "severity_weighted":
            if new_area_cells > 0:
                severity_score = 0.5 * float(severity_mean) + 0.5 * float(severity_max)
                r_severity = float(np.clip(0.75 * severity_score, 0.0, 0.75))
                reward += r_severity
                r_breakdown["r_severity"] += r_severity

        elif self.reward_profile == "exploration_balanced":
            view_area = max(float((self.vision_radius * 2 + 1) ** 2), 1.0)
            if new_area_cells > 0:
                r_area = float(np.clip(0.20 * float(new_area_cells) / view_area, 0.0, 0.10))
                reward += r_area
                r_breakdown["r_explore"] += r_area
            if cell_was_visited:
                repeat_penalty = -0.05
                reward += repeat_penalty
                r_breakdown["r_penalty"] += repeat_penalty

        return float(reward), r_breakdown

    def _update_discovered_boundary(self, pos: np.ndarray) -> Tuple[int, int]:
        new_area_cells = self._mark_visible_region(pos)
        visible_boundary = self._get_visible_boundary_points(pos)

        new_points = 0
        for bp in visible_boundary:
            if bp not in self.discovered_boundary:
                new_points += 1
            self.discovered_boundary.add(bp)
            self.confirmed_boundary_mask[bp[0], bp[1]] = True

        return new_points, new_area_cells

    def _check_boundary_in_vision(self, pos: np.ndarray) -> bool:
        return len(self._get_visible_boundary_points(pos)) > 0

    def _check_done(self) -> Tuple[bool, str]:
        coverage = self._boundary_coverage_rate()
        if self.curriculum_stage == 1:
            if any(self._check_boundary_in_vision(pos) for pos in self.drone_positions):
                return True, "mission_complete"
        elif self.curriculum_stage == 2:
            if coverage >= self.stage_targets[2]:
                return True, "mission_complete"
        else:
            if coverage >= self.stage_targets[3]:
                return True, "mission_complete"

        if self.step_count >= self.max_steps:
            return True, "max_steps_reached"
        if any(b <= 0 for b in self.drone_batteries):
            return True, "battery_depleted"
        return False, "ongoing"

    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        rewards = []
        step_reward_breakdown = {k: 0.0 for k in self.episode_reward_breakdown}
        prev_on_curve = self._discovered_on_current_boundary_count()

        n_act = len(actions)
        old_positions = [self.drone_positions[i].copy() for i in range(n_act)]
        new_positions = [self._execute_action(old_positions[i], actions[i]) for i in range(n_act)]

        for i, action in enumerate(actions):
            old_pos = old_positions[i]
            new_pos = new_positions[i]
            pre_boundary = len(self.discovered_boundary) == 0

            reward, r_breakdown = self._compute_reward(
                i, old_pos, new_pos, action, peer_new_positions=new_positions
            )

            self.drone_positions[i] = new_pos
            self.drone_momentums[i] = np.array(
                [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]],
                dtype=np.float32,
            )

            movement_dir = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
            if np.linalg.norm(movement_dir) > 1e-5:
                wind_effect = self.env_data.get_wind_effect(int(new_pos[0]), int(new_pos[1]), movement_dir)
                battery_cost = 1.0 + wind_effect["battery_penalty"] * 0.25
                self.drone_batteries[i] -= battery_cost
            else:
                self.drone_batteries[i] -= 0.1

            cell = (int(new_pos[0]), int(new_pos[1]))
            cell_was_visited = cell in self.visited_cells
            self.visited_cells.add(cell)

            heat_signal = self._get_heat_signal_features(new_pos)
            if self.first_heat_step < 0 and heat_signal["has_heat_signal"]:
                self.first_heat_step = self.step_count + 1

            severity_mean = 0.0
            severity_max = 0.0
            if self.reward_profile == "severity_weighted":
                _, severity_mean, severity_max = self._new_visible_area_severity_stats(new_pos)
            new_points, new_area_cells = self._update_discovered_boundary(new_pos)

            if pre_boundary and new_points == 0 and new_area_cells > 0:
                view_area = max(float((self.vision_radius * 2 + 1) ** 2), 1.0)
                r_area = self.pre_boundary_area_gain_weight * float(new_area_cells) / view_area
                r_area = float(np.clip(r_area, 0.0, self.pre_boundary_area_gain_clip))
                reward += r_area
                step_reward_breakdown["r_area_gain"] += r_area

            if new_points > 0 and self.first_boundary_step < 0:
                self.first_boundary_step = self.step_count + 1

            if new_points > 0:
                delta_coverage = new_points / max(self.total_boundary_points, 1)
                r_cov_gain = self.coverage_gain_weight * delta_coverage
                r_cov_gain = float(np.clip(r_cov_gain, 0.0, self.coverage_gain_clip))
                reward += r_cov_gain
                step_reward_breakdown["r_coverage_gain"] += r_cov_gain
                step_reward_breakdown["r_boundary"] += r_cov_gain

            profile_reward, profile_breakdown = self._compute_profile_reward(
                new_pos,
                cell_was_visited=cell_was_visited,
                new_area_cells=new_area_cells,
                severity_mean=severity_mean,
                severity_max=severity_max,
            )
            reward += profile_reward
            for key, value in profile_breakdown.items():
                step_reward_breakdown[key] += value

            self._recent_cells.append(cell)
            max_recent = max(1, self.pre_boundary_repeat_window * self.num_drones)
            if len(self._recent_cells) > max_recent:
                self._recent_cells = self._recent_cells[-max_recent:]

            for key, value in r_breakdown.items():
                step_reward_breakdown[key] += value

            rewards.append(float(reward))

        self.step_count += 1
        new_on_curve = self._discovered_on_current_boundary_count()
        self._coverage_gradient = (new_on_curve - prev_on_curve) / max(self.total_boundary_points, 1)

        if self.step_count % 20 == 0:
            new_boundary = self.env_data.detect_fire_boundary(
                time_step=self.step_count,
                start_sim_time=self.env_data.training_start_sim_time,
            )
            self.env_data.boundary_points = list(new_boundary)
            self.boundary_points = list(new_boundary)
            self.total_boundary_points = max(len(self.boundary_points), 1)
            self._build_boundary_set()
            if self.boundary_points:
                self.fire_centroid = np.mean(
                    np.array(self.boundary_points, dtype=np.float32), axis=0
                )
            self.env_data._compute_thermal_field()
            self._refresh_confirmed_boundary_state()

        done, done_reason = self._check_done()
        coverage = self._boundary_coverage_rate()
        timeout = done_reason == "max_steps_reached"
        zero_coverage_timeout = timeout and coverage <= 1e-9

        if done:
            if done_reason == "mission_complete":
                efficiency = 1.0 - np.clip(self.step_count / max(float(self.max_steps), 1.0), 0.0, 1.0)
                terminal_bonus = 6.0 + 4.0 * efficiency if self.curriculum_stage == 1 else 20.0 + 10.0 * efficiency
                rewards = [r + terminal_bonus / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] += terminal_bonus
            elif done_reason == "max_steps_reached":
                if self.curriculum_stage >= 2:
                    target = self.stage_targets[2] if self.curriculum_stage == 2 else self.stage_targets[3]
                    miss_gap = max(0.0, target - coverage)
                    terminal_penalty = 20.0 + 30.0 * miss_gap
                else:
                    early_gap = max(0.0, 0.02 - coverage) / 0.02
                    terminal_penalty = 10.0 + 20.0 * early_gap
                rewards = [r - terminal_penalty / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] -= terminal_penalty
            elif done_reason == "battery_depleted":
                terminal_penalty = 5.0
                rewards = [r - terminal_penalty / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] -= terminal_penalty

        for key in self.episode_reward_breakdown:
            self.episode_reward_breakdown[key] += step_reward_breakdown[key]

        info = {
            "step": self.step_count,
            "boundary_coverage": coverage,
            "observable_progress": coverage,
            "avg_distance_to_fire": float(
                np.mean([np.linalg.norm(pos - self.fire_centroid) for pos in self.drone_positions])
            ),
            "done_reason": done_reason,
            "scene_id": self.scene_id,
            "scene_key": self.scene_key,
            "observation_profile": self.observation_profile,
            "reward_profile": self.reward_profile,
            "vision_radius": self.vision_radius,
            "sensor_radius_cells": self.env_data.sensor_radius_cells,
            "max_steps": self.max_steps,
            "first_heat_step": int(self.first_heat_step),
            "first_boundary_step": int(self.first_boundary_step),
            "timeout": bool(timeout),
            "zero_coverage_timeout": bool(zero_coverage_timeout),
            "spawn_modes": list(self.spawn_modes),
            "reward_breakdown": self.episode_reward_breakdown.copy() if done else None,
            "stage_target": 0.0
            if self.curriculum_stage == 1
            else (self.stage_targets[2] if self.curriculum_stage == 2 else self.stage_targets[3]),
        }

        return self._get_observation(), rewards, done, info

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = int(stage)
        print(f"课程阶段已切换为 {self.curriculum_stage}")

    def get_env_info(self) -> Dict:
        return {
            "grid_size": self.grid_size,
            "total_boundary_points": self.total_boundary_points,
            "fire_centroid": self.fire_centroid.tolist(),
            "scene_id": self.scene_id,
            "scene_key": self.scene_key,
            "max_steps": self.max_steps,
            "max_battery": self.max_battery,
            "vision_radius": self.vision_radius,
            "sensor_radius_cells": self.env_data.sensor_radius_cells,
            "use_metadata_uav_params": self.use_metadata_uav_params,
            "observation_profile": self.observation_profile,
            "reward_profile": self.reward_profile,
            "stage2_target": self.stage_targets[2],
            "stage3_target": self.stage_targets[3],
            "init_area_percent": self.init_area_percent,
            "stage3_near_prob": self.stage3_near_prob,
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
        }


if __name__ == "__main__":
    env = FireSearchBaselineEnvironment()
    obs = env.reset()
    print([o.shape for o in obs["local_obs"]], obs["global_state"].shape)
    obs, rewards, done, info = env.step([0 for _ in range(env.num_drones)])
    print(rewards, done, info)
