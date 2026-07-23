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


class CooperativeTaskCoordinator:
    """Deployable-only task state for hierarchical two-drone coordination."""

    SEARCH = 0
    TRACK = 1
    REACQUIRE = 2
    ROLE_NAMES = ("SEARCH", "TRACK", "REACQUIRE")
    NUM_ROLES = 3
    CANDIDATE_FEATURES = 5
    ROLE_OBS_DIM = 27

    def __init__(
        self,
        env,
        peer_state_ttl: int,
        track_report_ttl: int,
        reacquire_report_ttl: int,
        role_decision_interval: int,
        role_min_dwell_steps: int,
    ):
        self.env = env
        self.peer_state_ttl = max(1, int(peer_state_ttl))
        self.track_report_ttl = max(1, int(track_report_ttl))
        self.reacquire_report_ttl = max(
            self.track_report_ttl + 1, int(reacquire_report_ttl)
        )
        self.role_decision_interval = max(1, int(role_decision_interval))
        self.role_min_dwell_steps = max(0, int(role_min_dwell_steps))
        self.reset()

    def reset(self) -> None:
        grid_size = self.env.grid_size
        num_drones = self.env.num_drones
        self.boundary_last_seen = [
            np.full(grid_size, -1, dtype=np.int32) for _ in range(num_drones)
        ]
        self.peer_messages = [None for _ in range(num_drones)]
        self.current_roles = [self.SEARCH for _ in range(num_drones)]
        self.role_start_steps = [0 for _ in range(num_drones)]
        self.assigned_targets = [None for _ in range(num_drones)]
        self.assigned_priorities = [0.0 for _ in range(num_drones)]
        self.assigned_task_valid = [False for _ in range(num_drones)]
        self.fire_ever_seen = [False for _ in range(num_drones)]
        self.candidates = np.zeros(
            (num_drones, self.NUM_ROLES, self.CANDIDATE_FEATURES),
            dtype=np.float32,
        )
        self.joint_role_mask = np.zeros(self.NUM_ROLES**2, dtype=np.int8)
        self.role_decision_required = True
        self.role_decision_agents = [True for _ in range(num_drones)]
        self.role_decision_reason = "reset"
        self.role_switch_count = 0
        self.pending_role_switch_count = 0
        self.invalid_role_count = 0
        self.task_conflict_count = 0
        self.expired_message_count = 0
        self.expired_boundary_report_count = 0

    def request_role_decision(
        self, reason: str, drone_idx: Optional[int] = None
    ) -> None:
        self.role_decision_required = True
        if drone_idx is None:
            self.role_decision_agents = [True for _ in range(self.env.num_drones)]
        else:
            self.role_decision_agents[int(drone_idx)] = True
        self.role_decision_reason = str(reason)

    def observe_boundary(
        self,
        drone_idx: int,
        visible_boundary: List[Tuple[int, int]],
        step: int,
    ) -> None:
        if not visible_boundary:
            return
        first_detection = not self.fire_ever_seen[drone_idx]
        self.fire_ever_seen[drone_idx] = True
        seen = self.boundary_last_seen[drone_idx]
        for y, x in visible_boundary:
            seen[int(y), int(x)] = int(step)
        if first_detection:
            self.request_role_decision(
                "local_fire_detected",
                None if self.env.communication_available[drone_idx] else drone_idx,
            )

    def expire_messages(self, step: int) -> None:
        for drone_idx, message in enumerate(self.peer_messages):
            if message is None:
                continue
            if int(step) - int(message["received_step"]) >= self.peer_state_ttl:
                self.peer_messages[drone_idx] = None
                self.expired_message_count += 1
                self.request_role_decision("peer_message_expired", drone_idx)

        for drone_idx, seen in enumerate(self.boundary_last_seen):
            stale = (seen >= 0) & ((int(step) - seen) >= self.reacquire_report_ttl)
            if np.any(stale):
                self.expired_boundary_report_count += int(np.count_nonzero(stale))
                seen[stale] = -1
                self.request_role_decision("boundary_report_expired", drone_idx)

    def sync_connected_agents(self, step: int) -> None:
        env = self.env
        if not env.communication_enabled or env.num_drones < 2:
            return
        for drone_idx in range(env.num_drones):
            peers = [
                peer
                for peer in range(env.num_drones)
                if peer != drone_idx
                and np.linalg.norm(
                    env.drone_positions[drone_idx] - env.drone_positions[peer]
                ) <= env.communication_radius
            ]
            if not peers:
                continue
            peer = min(
                peers,
                key=lambda idx: np.linalg.norm(
                    env.drone_positions[drone_idx] - env.drone_positions[idx]
                ),
            )
            previous = self.peer_messages[drone_idx]
            self.peer_messages[drone_idx] = {
                "peer": int(peer),
                "position": env.drone_positions[peer].copy(),
                "role": int(self.current_roles[peer]),
                "received_step": int(step),
                "source_step": int(step),
            }
            merged = np.maximum(
                self.boundary_last_seen[drone_idx],
                self.boundary_last_seen[peer],
            )
            self.boundary_last_seen[drone_idx] = merged.copy()
            self.boundary_last_seen[peer] = merged.copy()
            if self.fire_ever_seen[peer] and not self.fire_ever_seen[drone_idx]:
                self.fire_ever_seen[drone_idx] = True
                self.request_role_decision("fire_report_received")
            if previous is None:
                self.request_role_decision("communication_restored")

    def _directional_novelty(self, drone_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        env = self.env
        y, x = (int(v) for v in env.drone_positions[drone_idx])
        known = env.agent_known_masks[drone_idx]
        radius = max(1, env.vision_radius * 2)
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
        novelty = np.zeros(4, dtype=np.float32)
        for idx, (dy, dx) in enumerate(directions):
            cy = int(np.clip(y + dy * radius, 0, env.grid_size[0] - 1))
            cx = int(np.clip(x + dx * radius, 0, env.grid_size[1] - 1))
            y0, y1, x0, x1, mask = env._get_circular_window(cy, cx)
            patch = known[y0:y1, x0:x1]
            novelty[idx] = float(np.count_nonzero(~patch & mask) / max(np.count_nonzero(mask), 1))
        return directions, novelty

    def _candidate(self, drone_idx: int, role: int, step: int) -> np.ndarray:
        env = self.env
        pos = env.drone_positions[drone_idx].astype(np.float32)
        map_norm = max(float(np.linalg.norm(env.grid_size)), 1.0)
        result = np.zeros(self.CANDIDATE_FEATURES, dtype=np.float32)

        if role == self.SEARCH:
            directions, novelty = self._directional_novelty(drone_idx)
            best_candidates = np.flatnonzero(
                np.isclose(novelty, float(np.max(novelty)))
            )
            best = int(best_candidates[drone_idx % len(best_candidates)])
            direction = directions[best]
            distance = max(1, env.vision_radius * 2)
            target = np.array(
                [
                    np.clip(pos[0] + direction[0] * distance, 0, env.grid_size[0] - 1),
                    np.clip(pos[1] + direction[1] * distance, 0, env.grid_size[1] - 1),
                ],
                dtype=np.float32,
            )
            delta = (target - pos) / map_norm
            return np.array([1.0, delta[0], delta[1], novelty[best], 0.0], dtype=np.float32)

        seen = self.boundary_last_seen[drone_idx]
        ages = np.where(seen >= 0, int(step) - seen, np.iinfo(np.int32).max)
        if role == self.TRACK:
            valid = (seen >= 0) & (ages < self.track_report_ttl)
            ttl = self.track_report_ttl
        else:
            valid = (
                self.fire_ever_seen[drone_idx]
                & (seen >= 0)
                & (ages >= self.track_report_ttl)
                & (ages < self.reacquire_report_ttl)
            )
            ttl = self.reacquire_report_ttl
        points = np.argwhere(valid)
        if points.size == 0:
            return result
        distances = np.linalg.norm(points.astype(np.float32) - pos[None, :], axis=1)
        point_ages = ages[valid].astype(np.float32)
        priorities = point_ages / max(float(ttl), 1.0) - distances / map_norm
        best = int(np.argmax(priorities))
        target = points[best].astype(np.float32)
        delta = (target - pos) / map_norm
        return np.array(
            [
                1.0,
                delta[0],
                delta[1],
                float(np.clip(priorities[best] + 1.0, 0.0, 1.0)),
                float(np.clip(point_ages[best] / max(float(ttl), 1.0), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def refresh_candidates(self, step: int) -> None:
        self.expire_messages(step)
        for drone_idx in range(self.env.num_drones):
            for role in range(self.NUM_ROLES):
                self.candidates[drone_idx, role] = self._candidate(
                    drone_idx, role, step
                )
        self._refresh_joint_mask()

    def _refresh_joint_mask(self) -> None:
        if self.env.num_drones != 2:
            raise ValueError("hierarchical role matching currently requires exactly two drones")
        mask = np.zeros((self.NUM_ROLES, self.NUM_ROLES), dtype=np.int8)
        for role0 in range(self.NUM_ROLES):
            for role1 in range(self.NUM_ROLES):
                if not self.candidates[0, role0, 0] or not self.candidates[1, role1, 0]:
                    continue
                if (
                    not self.role_decision_agents[0]
                    and role0 != self.current_roles[0]
                ):
                    continue
                if (
                    not self.role_decision_agents[1]
                    and role1 != self.current_roles[1]
                ):
                    continue
                jointly_connected = bool(
                    self.env.communication_available[0]
                    and self.env.communication_available[1]
                )
                if jointly_connected and role0 == self.TRACK and role1 == self.TRACK:
                    target0 = self._absolute_candidate_target(0, role0)
                    target1 = self._absolute_candidate_target(1, role1)
                    if np.linalg.norm(target0 - target1) < self.env.vision_radius:
                        continue
                mask[role0, role1] = 1
        if not np.any(mask):
            mask[self.SEARCH, self.SEARCH] = 1
        self.joint_role_mask = mask.reshape(-1)

    def _absolute_candidate_target(self, drone_idx: int, role: int) -> np.ndarray:
        candidate = self.candidates[drone_idx, role]
        map_norm = max(float(np.linalg.norm(self.env.grid_size)), 1.0)
        return self.env.drone_positions[drone_idx] + candidate[1:3] * map_norm

    def role_observations(self, step: int) -> List[np.ndarray]:
        observations = []
        for drone_idx in range(self.env.num_drones):
            current_role = np.zeros(self.NUM_ROLES, dtype=np.float32)
            current_role[self.current_roles[drone_idx]] = 1.0
            message = self.peer_messages[drone_idx]
            peer_valid = float(message is not None)
            peer_age = 0.0
            peer_role = np.zeros(self.NUM_ROLES, dtype=np.float32)
            if message is not None:
                peer_age = float(
                    np.clip(
                        (int(step) - int(message["received_step"])) / self.peer_state_ttl,
                        0.0,
                        1.0,
                    )
                )
                peer_role[int(message["role"])] = 1.0
            extras = np.concatenate(
                [
                    np.array(
                        [
                            self.env.drone_batteries[drone_idx] / self.env.max_battery,
                        ],
                        dtype=np.float32,
                    ),
                    current_role,
                    np.array(
                        [
                            np.clip(
                                (int(step) - self.role_start_steps[drone_idx])
                                / max(float(self.role_decision_interval), 1.0),
                                0.0,
                                1.0,
                            ),
                            float(self.env.communication_available[drone_idx]),
                            peer_valid,
                            peer_age,
                        ],
                        dtype=np.float32,
                    ),
                    peer_role,
                    np.array([float(self.fire_ever_seen[drone_idx])], dtype=np.float32),
                ]
            )
            role_obs = np.concatenate([self.candidates[drone_idx].reshape(-1), extras])
            if role_obs.shape[0] != self.ROLE_OBS_DIM:
                raise RuntimeError(
                    f"role observation dimension mismatch: {role_obs.shape[0]} != {self.ROLE_OBS_DIM}"
                )
            observations.append(role_obs.astype(np.float32))
        return observations

    def motion_features(self, drone_idx: int) -> List[float]:
        role = self.current_roles[drone_idx]
        role_one_hot = [float(role == idx) for idx in range(self.NUM_ROLES)]
        candidate = self.candidates[drone_idx, role]
        return role_one_hot + [
            float(candidate[1]),
            float(candidate[2]),
            float(candidate[0]),
            float(candidate[3]),
        ]

    def apply_joint_roles(self, roles: List[int], step: int) -> None:
        if len(roles) != 2:
            raise ValueError("joint role assignment must contain two roles")
        flat_index = int(roles[0]) * self.NUM_ROLES + int(roles[1])
        if flat_index < 0 or flat_index >= self.joint_role_mask.size or not self.joint_role_mask[flat_index]:
            self.invalid_role_count += 1
            raise ValueError(f"invalid joint role assignment: {roles}")
        for drone_idx, role in enumerate(roles):
            role = int(role)
            if role != self.current_roles[drone_idx]:
                self.role_switch_count += 1
                self.pending_role_switch_count += 1
                self.role_start_steps[drone_idx] = int(step)
            self.current_roles[drone_idx] = role
            self.assigned_targets[drone_idx] = self._absolute_candidate_target(
                drone_idx, role
            )
            self.assigned_priorities[drone_idx] = float(
                self.candidates[drone_idx, role, 3]
            )
            self.assigned_task_valid[drone_idx] = bool(
                self.candidates[drone_idx, role, 0]
            )
        self.role_decision_required = False
        self.role_decision_agents = [False for _ in range(self.env.num_drones)]
        self.role_decision_reason = "assigned"

    def maybe_request_periodic_decision(self, step: int) -> None:
        if int(step) > 0 and int(step) % self.role_decision_interval == 0:
            if all(
                int(step) - started >= self.role_min_dwell_steps
                for started in self.role_start_steps
            ):
                self.request_role_decision("periodic_review")


class FireSearchBaselineEnvironment(gym.Env):
    STAGE1_TRACK_WINDOW = 15
    STAGE1_TRACK_REQUIRED = 11
    STAGE1_MIN_UNIQUE_BOUNDARY = 8
    STAGE1_MIN_POST_CONTACT_PROGRESS = 4
    STAGE2_TRACK_WINDOW = 10
    STAGE2_TRACK_REQUIRED = 8
    STAGE2_REACQUIRE_MIN_VISIBLE = 5
    """Baseline multi-drone fire boundary search environment."""

    TERMINATION_MODES = {"target_stop", "full_horizon", "post_target_train"}

    OBSERVATION_PROFILE_DIMS = {
        "baseline": {"local_obs_dim": 17, "global_state_dim": 19},
        "static_terrain": {"local_obs_dim": 24, "global_state_dim": 19},
        "dynamic_front": {"local_obs_dim": 23, "global_state_dim": 19},
        "risk_aware": {"local_obs_dim": 20, "global_state_dim": 19},
        "cooperative_exploration": {"local_obs_dim": 24, "global_state_dim": 19},
        "persistent_cooperative": {"local_obs_dim": 24, "global_state_dim": 19},
    }
    REWARD_PROFILES = {
        "boundary_coverage",
        "front_detection",
        "severity_weighted",
        "exploration_balanced",
        "novelty_search",
        "persistent_boundary",
    }
    REWARD_BREAKDOWN_KEYS = [
        "r_discover",
        "r_coverage_gain",
        "r_area_gain",
        "r_boundary",
        "r_front",
        "r_severity",
        "r_explore",
        "r_search",
        "r_contact",
        "r_track",
        "r_reacquire",
        "r_novelty",
        "r_revisit",
        "r_invalid",
        "r_overlap",
        "r_penalty",
        "r_terminal",
        "r_milestone",
        "r_hold",
        "r_tail",
        "r_fresh_gain",
        "r_assigned_gain",
        "r_role_switch",
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
        termination_mode: str = "target_stop",
        post_target_goal: float = 0.60,
        post_target_step_penalty: float = -0.005,
        post_target_step_cost_fraction: float = 0.25,
        post_target_hold_weight: float = 0.10,
        post_target_tail_weight: float = 20.0,
        post_target_milestone_70: float = 5.0,
        post_target_milestone_80: float = 10.0,
        post_target_extra_steps: Optional[int] = None,
        communication_enabled: bool = False,
        communication_radius_factor: float = 4.0,
        action_mask_enabled: bool = False,
        novelty_reward_weight: float = 0.12,
        novelty_step_penalty: float = -0.04,
        novelty_revisit_penalty: float = 0.08,
        invalid_action_penalty: float = 0.25,
        team_overlap_penalty: float = 0.05,
        hierarchical_roles_enabled: bool = False,
        peer_state_ttl: int = 8,
        track_report_ttl: int = 20,
        reacquire_report_ttl: int = 60,
        role_decision_interval: int = 20,
        role_min_dwell_steps: int = 10,
        boundary_match_radius: int = 2,
        boundary_freshness_tau: float = 40.0,
        fresh_coverage_gain_weight: float = 20.0,
        assigned_boundary_gain_weight: float = 6.0,
        role_switch_penalty: float = 0.02,
        mask_thermal_below_signal: bool = False,
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
        self.termination_mode = self._validate_termination_mode(termination_mode)
        self.curriculum_stage = int(curriculum_stage)
        self.mode = mode
        self.fixed_scene_key = fixed_scene_key
        self.scene_keys = [str(key) for key in scene_keys] if scene_keys is not None else None
        area_percent = init_area_percent if init_area_percent is not None else init_percentile
        self.init_area_percent = None if area_percent is None else float(area_percent)
        self.init_percentile = self.init_area_percent
        self.stage_targets = {2: float(stage2_target), 3: float(stage3_target)}
        self.stage3_near_prob = float(stage3_near_prob)
        self.curriculum_substage = "1"
        self.post_target_goal = float(np.clip(post_target_goal, 0.20, 0.80))
        self.post_target_step_penalty = float(post_target_step_penalty)
        self.post_target_step_cost_fraction = max(
            0.0, float(post_target_step_cost_fraction)
        )
        self.post_target_hold_weight = float(post_target_hold_weight)
        self.post_target_tail_weight = float(post_target_tail_weight)
        self.post_target_milestone_70 = float(post_target_milestone_70)
        self.post_target_milestone_80 = float(post_target_milestone_80)
        self.post_target_extra_steps = (
            None if post_target_extra_steps is None else max(1, int(post_target_extra_steps))
        )
        self.communication_enabled = bool(communication_enabled)
        self.communication_radius_factor = max(0.0, float(communication_radius_factor))
        self.action_mask_enabled = bool(action_mask_enabled)
        self.novelty_reward_weight = max(0.0, float(novelty_reward_weight))
        self.novelty_step_penalty = float(novelty_step_penalty)
        self.novelty_revisit_penalty = max(0.0, float(novelty_revisit_penalty))
        self.invalid_action_penalty = max(0.0, float(invalid_action_penalty))
        self.team_overlap_penalty = max(0.0, float(team_overlap_penalty))
        self.hierarchical_roles_enabled = bool(hierarchical_roles_enabled)
        if self.hierarchical_roles_enabled and self.num_drones != 2:
            raise ValueError("hierarchical_roles_enabled currently requires num_drones=2")
        if (
            self.observation_profile == "persistent_cooperative"
            or self.reward_profile == "persistent_boundary"
        ) and not self.hierarchical_roles_enabled:
            raise ValueError(
                "persistent cooperative profiles require hierarchical_roles_enabled"
            )
        if (
            self.reward_profile == "persistent_boundary"
            and self.observation_profile != "persistent_cooperative"
        ):
            raise ValueError(
                "persistent_boundary requires observation_profile='persistent_cooperative'"
            )
        self.peer_state_ttl = max(1, int(peer_state_ttl))
        self.track_report_ttl = max(1, int(track_report_ttl))
        self.reacquire_report_ttl = max(
            self.track_report_ttl + 1, int(reacquire_report_ttl)
        )
        self.role_decision_interval = max(1, int(role_decision_interval))
        self.role_min_dwell_steps = max(0, int(role_min_dwell_steps))
        self.boundary_match_radius = max(0, int(boundary_match_radius))
        self.boundary_freshness_tau = max(1.0, float(boundary_freshness_tau))
        self.fresh_coverage_gain_weight = max(0.0, float(fresh_coverage_gain_weight))
        self.assigned_boundary_gain_weight = max(
            0.0, float(assigned_boundary_gain_weight)
        )
        self.role_switch_penalty = max(0.0, float(role_switch_penalty))
        self.mask_thermal_below_signal = bool(mask_thermal_below_signal)

        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)

        self.coverage_gain_weight = 40.0
        self.coverage_gain_clip = 2.0
        self.stage1_explore_reward_cap = 25.0
        self.pre_boundary_area_gain_weight = 0.35
        self.pre_boundary_area_gain_clip = 0.08
        self.pre_boundary_repeat_window = 12
        self.pre_boundary_repeat_penalty = 0.04
        self.zero_coverage_timeout_extra_penalty = 25.0

        scene_keys_by_split = None
        if self.scene_keys is not None:
            split = SceneManager(data_dir).dataset_index.normalize_mode(mode)
            scene_keys_by_split = {split: self.scene_keys}
        self.scene_manager = SceneManager(data_dir, scene_keys_by_split=scene_keys_by_split)
        self._load_new_scene()

        dims = self.OBSERVATION_PROFILE_DIMS[self.observation_profile]
        self.local_obs_dim = dims["local_obs_dim"]
        self.global_state_dim = dims["global_state_dim"]
        observation_spaces = {
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
                "action_masks": spaces.Tuple(
                    tuple(
                        spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)
                        for _ in range(self.num_drones)
                    )
                ),
            }
        if self.hierarchical_roles_enabled:
            observation_spaces.update(
                {
                    "role_obs": spaces.Tuple(
                        tuple(
                            spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(CooperativeTaskCoordinator.ROLE_OBS_DIM,),
                                dtype=np.float32,
                            )
                            for _ in range(self.num_drones)
                        )
                    ),
                    "joint_role_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(CooperativeTaskCoordinator.NUM_ROLES**2,),
                        dtype=np.int8,
                    ),
                    "role_decision_required": spaces.Discrete(2),
                }
            )
        self.observation_space = spaces.Dict(observation_spaces)

        self.max_battery = int(self.max_steps * 2.0)
        self.step_count = 0
        self.drone_positions: List[np.ndarray] = []
        self.drone_batteries: List[float] = []
        self.drone_momentums: List[np.ndarray] = []
        self.visited_cells = set()
        self.discovered_boundary = set()
        self.discovered_front = set()
        self.discovered_area_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.agent_observed_masks = [
            np.zeros(self.grid_size, dtype=np.bool_) for _ in range(self.num_drones)
        ]
        self.agent_known_masks = [mask.copy() for mask in self.agent_observed_masks]
        self.communication_available = [False] * self.num_drones
        self.communication_agent_steps = 0
        self.shared_new_cells = 0
        self.pre_boundary_agent_steps = 0
        self.pre_boundary_revisit_steps = 0
        self.team_overlap_sum = 0.0
        self.invalid_action_count = 0
        self.confirmed_boundary_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.boundary_ever_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.boundary_last_seen_step = np.full(self.grid_size, -1, dtype=np.int32)
        self._mark_current_boundary_ever()
        self._coverage_gradient = 0.0
        self._episode_explore_reward_total = 0.0
        self.first_heat_step = -1
        self.first_boundary_step = -1
        self.stage1_tracking_window: List[bool] = []
        self.stage1_contact_boundary_cells = 0
        self.stage1_tracking_success = False
        self.stage2_tracking_window: List[bool] = []
        self.stage2_lost_after_contact = False
        self.stage2_reacquire_success = False
        self.stage2_contact_success = False
        self.target_reached = False
        self.first_target_step = -1
        self.boundary_refreshed = False
        self.coverage_before_boundary_refresh = 0.0
        self.coverage_after_boundary_refresh = 0.0
        self.coverage_history: List[float] = []
        self.fresh_coverage_history: List[float] = []
        self.tolerant_coverage_history: List[float] = []
        self.coverage_action_gain = 0.0
        self.coverage_refresh_drop = 0.0
        self.major_refresh_count = 0
        self.refresh_recovery_successes = 0
        self.refresh_recovery_times: List[int] = []
        self._pending_refresh_step = -1
        self._pending_recovery_target = 0.0
        self.post_target_milestones = {"0.60": False, "0.70": False, "0.80": False}
        self._recent_cells: List[Tuple[int, int]] = []

        self.episode_reward_breakdown = self._empty_reward_breakdown()
        self.task_coordinator = (
            CooperativeTaskCoordinator(
                self,
                peer_state_ttl=self.peer_state_ttl,
                track_report_ttl=self.track_report_ttl,
                reacquire_report_ttl=self.reacquire_report_ttl,
                role_decision_interval=self.role_decision_interval,
                role_min_dwell_steps=self.role_min_dwell_steps,
            )
            if self.hierarchical_roles_enabled
            else None
        )

        print(
            "基线环境已初始化 | "
            f"模式={mode} | 本地观测维度={self.local_obs_dim} | "
            f"全局状态维度={self.global_state_dim} | "
            f"observation_profile={self.observation_profile} | "
            f"reward_profile={self.reward_profile} | "
            f"termination_mode={self.termination_mode} | "
            f"communication={self.communication_enabled} | "
            f"action_mask={self.action_mask_enabled}"
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

    @classmethod
    def _validate_termination_mode(cls, mode: str) -> str:
        mode = str(mode).lower()
        if mode not in cls.TERMINATION_MODES:
            raise ValueError(
                f"Unknown termination_mode {mode!r}. "
                f"Expected one of: {sorted(cls.TERMINATION_MODES)}"
            )
        return mode

    def _empty_reward_breakdown(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.REWARD_BREAKDOWN_KEYS}

    def _boundary_coverage_gain_reward(self, new_points: int) -> float:
        delta_coverage = float(new_points) / max(float(self.total_boundary_points), 1.0)
        weights = {1: 12.0, 2: 6.0, 3: 30.0, 4: 30.0}
        clips = {1: 0.60, 2: 0.30, 3: 1.50, 4: 1.50}
        stage = int(getattr(self, "curriculum_stage", 1))
        reward = weights.get(stage, self.coverage_gain_weight) * delta_coverage
        return float(np.clip(reward, 0.0, clips.get(stage, self.coverage_gain_clip)))

    def _pre_boundary_area_reward(self, new_area_cells: int) -> float:
        view_area = max(float((self.vision_radius * 2 + 1) ** 2), 1.0)
        reward = self.pre_boundary_area_gain_weight * float(new_area_cells) / view_area
        return float(np.clip(reward, 0.0, self.pre_boundary_area_gain_clip))

    def _timeout_terminal_penalty(self, coverage: float) -> float:
        stage = int(self.curriculum_stage)
        if stage == 1:
            return 3.0
        if stage == 2:
            penalty = 3.0 + (1.0 if float(coverage) <= 1e-9 else 0.0)
            return float(min(5.0, penalty))
        target = float(self.stage_targets[3])
        miss_gap = max(0.0, target - float(coverage))
        penalty = 4.0 + 3.0 * miss_gap
        if float(coverage) <= 1e-9:
            penalty += 1.0
        return float(min(6.0, penalty))

    def _discovered_on_current_boundary_count(self) -> int:
        return sum(1 for p in self.discovered_boundary if p in self._boundary_set)

    def _boundary_coverage_rate(self) -> float:
        return self._discovered_on_current_boundary_count() / max(self.total_boundary_points, 1)

    def _mark_current_boundary_ever(self):
        for by, bx in self.boundary_points:
            self.boundary_ever_mask[int(by), int(bx)] = True

    def _historical_boundary_union_coverage_rate(self) -> float:
        total = int(np.count_nonzero(self.boundary_ever_mask))
        if total == 0:
            return 0.0
        observed = int(np.count_nonzero(self.confirmed_boundary_mask & self.boundary_ever_mask))
        return float(observed / total)

    def _boundary_freshness_metrics(self) -> Tuple[float, float]:
        if not self.boundary_points:
            return 0.0, 0.0
        tolerant_sum = 0.0
        fresh_sum = 0.0
        radius = self.boundary_match_radius
        height, width = self.grid_size
        for by, bx in self.boundary_points:
            y = int(by)
            x = int(bx)
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            patch = self.boundary_last_seen_step[y0:y1, x0:x1]
            valid_steps = patch[patch >= 0]
            if valid_steps.size == 0:
                continue
            newest = int(np.max(valid_steps))
            age = int(self.step_count) - newest
            if age < 0:
                continue
            tolerant_sum += 1.0
            fresh_sum += float(np.exp(-age / self.boundary_freshness_tau))
        total = max(float(len(self.boundary_points)), 1.0)
        return float(tolerant_sum / total), float(fresh_sum / total)

    def _mark_boundary_seen(
        self,
        drone_idx: int,
        visible_boundary: List[Tuple[int, int]],
        step: int,
    ) -> None:
        for y, x in visible_boundary:
            self.boundary_last_seen_step[int(y), int(x)] = int(step)
        if self.task_coordinator is not None:
            self.task_coordinator.observe_boundary(
                drone_idx,
                visible_boundary,
                step,
            )

    def _stage_target_reached(self, coverage: float) -> bool:
        if self.curriculum_stage == 1:
            return bool(getattr(self, "stage1_tracking_success", False))
        if self.curriculum_stage == 2:
            return bool(getattr(self, "stage2_contact_success", False))
        target = self.stage_targets[3]
        return float(coverage) >= float(target)

    def _objective_coverage_rate(self) -> float:
        if getattr(self, "reward_profile", "boundary_coverage") == "persistent_boundary":
            return self._boundary_freshness_metrics()[1]
        return self._boundary_coverage_rate()

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

    @property
    def communication_radius(self) -> float:
        return float(self.communication_radius_factor * self.vision_radius)

    def _mark_agent_visible_region(self, drone_idx: int, pos: np.ndarray) -> int:
        y, x = int(pos[0]), int(pos[1])
        y_min, y_max, x_min, x_max, local_mask = self._get_circular_window(y, x)
        own_patch = self.agent_observed_masks[drone_idx][y_min:y_max, x_min:x_max]
        newly_visible = int(np.count_nonzero(~own_patch[local_mask]))
        own_patch[local_mask] = True
        known_patch = self.agent_known_masks[drone_idx][y_min:y_max, x_min:x_max]
        known_patch[local_mask] = True
        return newly_visible

    def _sync_exploration_knowledge(self, count_metrics: bool = True) -> None:
        for drone_idx in range(self.num_drones):
            self.agent_known_masks[drone_idx] |= self.agent_observed_masks[drone_idx]
        self.communication_available = [False] * self.num_drones
        shared_cells = 0

        if self.communication_enabled and self.num_drones > 1 and self.communication_radius > 0.0:
            remaining = set(range(self.num_drones))
            while remaining:
                component = {remaining.pop()}
                frontier = list(component)
                while frontier:
                    current = frontier.pop()
                    connected = {
                        peer
                        for peer in list(remaining)
                        if np.linalg.norm(
                            self.drone_positions[current] - self.drone_positions[peer]
                        ) <= self.communication_radius
                    }
                    remaining.difference_update(connected)
                    component.update(connected)
                    frontier.extend(connected)

                if len(component) <= 1:
                    continue
                merged = np.zeros(self.grid_size, dtype=np.bool_)
                for drone_idx in component:
                    merged |= self.agent_known_masks[drone_idx]
                    merged |= self.agent_observed_masks[drone_idx]
                for drone_idx in component:
                    shared_cells += int(
                        np.count_nonzero(merged & ~self.agent_known_masks[drone_idx])
                    )
                    self.agent_known_masks[drone_idx] = merged.copy()
                    self.communication_available[drone_idx] = True

        if count_metrics:
            self.communication_agent_steps += sum(self.communication_available)
            self.shared_new_cells += shared_cells

    def _cooperative_exploration_features(
        self, drone_idx: int, y: int, x: int
    ) -> List[float]:
        known = self.agent_known_masks[drone_idx]
        radius = max(1, self.vision_radius * 2)
        y_min = max(0, y - radius)
        y_max = min(self.grid_size[0], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.grid_size[1], x + radius + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dy = yy - y
        dx = xx - x
        circle = dy * dy + dx * dx <= radius * radius
        abs_dy = np.abs(dy)
        abs_dx = np.abs(dx)
        sectors = [
            circle & (dy < 0) & (abs_dy >= abs_dx),
            circle & (dy > 0) & (abs_dy >= abs_dx),
            circle & (dx < 0) & (abs_dx > abs_dy),
            circle & (dx > 0) & (abs_dx > abs_dy),
        ]
        known_patch = known[y_min:y_max, x_min:x_max]
        novelty = [
            float(np.count_nonzero(~known_patch & sector) / max(np.count_nonzero(sector), 1))
            for sector in sectors
        ]

        peer_dy = 0.0
        peer_dx = 0.0
        if self.communication_available[drone_idx]:
            connected = [
                peer
                for peer in range(self.num_drones)
                if peer != drone_idx
                and np.linalg.norm(
                    self.drone_positions[drone_idx] - self.drone_positions[peer]
                ) <= self.communication_radius
            ]
            if connected:
                peer = min(
                    connected,
                    key=lambda idx: np.linalg.norm(
                        self.drone_positions[drone_idx] - self.drone_positions[idx]
                    ),
                )
                delta = (
                    self.drone_positions[peer] - self.drone_positions[drone_idx]
                ) / max(self.communication_radius, 1.0)
                peer_dy = float(np.clip(delta[0], -1.0, 1.0))
                peer_dx = float(np.clip(delta[1], -1.0, 1.0))

        return novelty + [
            peer_dy,
            peer_dx,
            float(self.communication_available[drone_idx]),
        ]

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
        self.agent_observed_masks = [
            np.zeros(self.grid_size, dtype=np.bool_) for _ in range(self.num_drones)
        ]
        self.agent_known_masks = [mask.copy() for mask in self.agent_observed_masks]
        self.communication_available = [False] * self.num_drones
        self.communication_agent_steps = 0
        self.shared_new_cells = 0
        self.pre_boundary_agent_steps = 0
        self.pre_boundary_revisit_steps = 0
        self.team_overlap_sum = 0.0
        self.invalid_action_count = 0
        self.confirmed_boundary_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.boundary_ever_mask = np.zeros(self.grid_size, dtype=np.bool_)
        self.boundary_last_seen_step = np.full(
            self.grid_size, -1, dtype=np.int32
        )
        self._mark_current_boundary_ever()
        self._coverage_gradient = 0.0
        self._episode_explore_reward_total = 0.0
        self.first_heat_step = -1
        self.first_boundary_step = -1
        self.stage1_tracking_window = []
        self.stage1_contact_boundary_cells = 0
        self.stage1_tracking_success = False
        self.stage2_tracking_window = []
        self.stage2_lost_after_contact = False
        self.stage2_reacquire_success = False
        self.stage2_contact_success = False
        self.target_reached = False
        self.first_target_step = -1
        self.boundary_refreshed = False
        self.coverage_before_boundary_refresh = 0.0
        self.coverage_after_boundary_refresh = 0.0
        self.coverage_history = []
        self.fresh_coverage_history = []
        self.tolerant_coverage_history = []
        self.coverage_action_gain = 0.0
        self.coverage_refresh_drop = 0.0
        self.major_refresh_count = 0
        self.refresh_recovery_successes = 0
        self.refresh_recovery_times = []
        self._pending_refresh_step = -1
        self._pending_recovery_target = 0.0
        self.post_target_milestones = {"0.60": False, "0.70": False, "0.80": False}
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

        if (
            self.observation_profile
            in {"cooperative_exploration", "persistent_cooperative"}
            or self.communication_enabled
        ):
            for drone_idx, pos in enumerate(self.drone_positions):
                self._mark_agent_visible_region(drone_idx, pos)
            if self.reward_profile == "novelty_search":
                for mask in self.agent_observed_masks:
                    self.discovered_area_mask |= mask
            self._sync_exploration_knowledge(count_metrics=False)

        if self.task_coordinator is not None:
            self.task_coordinator.reset()
            self.task_coordinator.sync_connected_agents(self.step_count)

        if np.any(self.boundary_last_seen_step != -1):
            raise RuntimeError("episode reset left stale boundary timestamps")

        return self._get_observation()

    def _spawn_randomly(self, drone_idx: int) -> np.ndarray:
        min_factor, max_factor, mode = self._spawn_distance_profile()
        pos = self._spawn_by_boundary_distance(min_factor, max_factor)
        self.spawn_modes.append(mode)
        return pos

    def _spawn_distance_profile(self) -> Tuple[float, Optional[float], str]:
        if self.curriculum_stage == 1:
            return 1.0, 2.0, "stage1_contact"
        if self.curriculum_stage == 2:
            profiles = {
                "2A": (1.5, 3.0, "stage2a"),
                "2B": (2.5, 5.0, "stage2b"),
                "2C": (2.5, None, "stage2c"),
            }
            return profiles.get(self.curriculum_substage, profiles["2A"])
        return 2.5, None, "far"

    def _spawn_by_boundary_distance(
        self,
        min_factor: float,
        max_factor: Optional[float],
    ) -> np.ndarray:
        h, w = self.grid_size
        margin = max(2, self.vision_radius // 2)
        boundary = np.asarray(self.boundary_points, dtype=np.float32)
        if boundary.size == 0:
            raise RuntimeError("Cannot sample a curriculum spawn without fire boundary points")

        min_distance = float(min_factor) * self.vision_radius
        max_distance = None if max_factor is None else float(max_factor) * self.vision_radius
        candidates = []
        for _ in range(200):
            pos = np.array(
                [
                    np.random.randint(margin, h - margin),
                    np.random.randint(margin, w - margin),
                ],
                dtype=np.float32,
            )
            nearest = float(np.min(np.linalg.norm(boundary - pos, axis=1)))
            if nearest < min_distance or (max_distance is not None and nearest > max_distance):
                continue
            if self._too_close_to_existing_drones(pos):
                continue
            fire_info = self.env_data.get_local_fire_info(
                int(pos[0]), int(pos[1]), self.vision_radius
            )
            if fire_info.get("fire_count", 0) > 0 or fire_info.get("boundary_count", 0) > 0:
                continue
            return pos

        for y in range(margin, h - margin):
            for x in range(margin, w - margin):
                pos = np.array([y, x], dtype=np.float32)
                nearest = float(np.min(np.linalg.norm(boundary - pos, axis=1)))
                if nearest < min_distance or (max_distance is not None and nearest > max_distance):
                    continue
                if self._too_close_to_existing_drones(pos):
                    continue
                fire_info = self.env_data.get_local_fire_info(y, x, self.vision_radius)
                if fire_info.get("fire_count", 0) == 0 and fire_info.get("boundary_count", 0) == 0:
                    candidates.append(pos)
        if not candidates:
            raise RuntimeError(
                f"No legal spawn for stage={self.curriculum_stage} "
                f"substage={self.curriculum_substage} distance=[{min_factor}, {max_factor}]R"
            )
        return candidates[np.random.randint(0, len(candidates))]

    def _too_close_to_existing_drones(self, pos: np.ndarray) -> bool:
        min_spacing = float(self.vision_radius * 0.8)
        return any(np.linalg.norm(pos - other_pos) < min_spacing for other_pos in self.drone_positions)

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
        if self.task_coordinator is not None:
            self.task_coordinator.refresh_candidates(self.step_count)
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
            if self.mask_thermal_below_signal:
                curr_heat = max(0.0, self.env_data.get_thermal_value(y, x))
                thermal_valid = bool(
                    local_fire_info.get("fire_count", 0) > 0 or curr_heat >= 0.50
                )
                if not thermal_valid:
                    grad_y, grad_x = 0.0, 0.0

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
            elif self.observation_profile == "cooperative_exploration":
                local_obs.extend(self._cooperative_exploration_features(i, y, x))
            elif self.observation_profile == "persistent_cooperative":
                if self.task_coordinator is None:
                    raise RuntimeError(
                        "persistent_cooperative requires hierarchical_roles_enabled"
                    )
                local_obs.extend(self.task_coordinator.motion_features(i))
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
            float(self.termination_mode == "post_target_train"),
            float(self._coverage_gradient),
            undiscovered_density,
        ]

        observation = {
            "local_obs": local_obs_list,
            "global_state": np.array(global_state, dtype=np.float32),
            "action_masks": self._get_action_masks(),
        }
        if self.task_coordinator is not None:
            observation.update(
                {
                    "role_obs": self.task_coordinator.role_observations(
                        self.step_count
                    ),
                    "joint_role_mask": self.task_coordinator.joint_role_mask.copy(),
                    "role_decision_required": int(
                        self.task_coordinator.role_decision_required
                    ),
                }
            )
        return observation

    def _get_action_masks(self) -> List[np.ndarray]:
        if not self.action_mask_enabled:
            return [np.ones(self.num_actions, dtype=np.int8) for _ in self.drone_positions]
        masks = []
        for pos in self.drone_positions:
            y, x = int(pos[0]), int(pos[1])
            masks.append(
                np.array(
                    [
                        x < self.grid_size[1] - 1,
                        x > 0,
                        y > 0,
                        y < self.grid_size[0] - 1,
                        True,
                    ],
                    dtype=np.int8,
                )
            )
        return masks

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
        """分层热信号判定，替代旧的全局阈值检测。

        新的信号层级：
        - local_fire_visible: 视野内真实火点数 > 0
        - thermal_sensor_signal: 当前位置 thermal_potential >= 0.50
        - has_heat_signal: 以上任一为 True
        """
        y, x = int(pos[0]), int(pos[1])
        curr_heat = max(0.0, self.env_data.get_thermal_value(y, x))
        local_fire_info = self.env_data.get_local_fire_info(y, x, self.vision_radius)
        local_fire_visible = local_fire_info.get("fire_count", 0) > 0
        thermal_sensor_signal = curr_heat >= 0.50
        has_heat_signal = local_fire_visible or thermal_sensor_signal
        return {
            "current_heat": float(curr_heat),
            "has_heat_signal": bool(has_heat_signal),
            "local_fire_visible": bool(local_fire_visible),
            "thermal_sensor_signal": bool(thermal_sensor_signal),
        }

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
                r_disc = 0.30
            elif self.curriculum_stage == 2:
                r_disc = 0.15
            else:
                r_disc = 0.10
            reward += r_disc
            r_breakdown["r_discover"] += r_disc
            r_breakdown["r_boundary"] += r_disc

        if self.termination_mode == "post_target_train" and self.target_reached:
            step_penalty = self._effective_post_target_step_penalty()
        elif getattr(self, "reward_profile", "boundary_coverage") == "novelty_search" and len(self.discovered_boundary) == 0:
            step_penalty = self.novelty_step_penalty
        else:
            step_penalty = -0.005 if self.curriculum_stage == 1 else -0.01
        reward += step_penalty
        r_breakdown["r_penalty"] += step_penalty

        invalid_action = int(action) != 4 and np.array_equal(new_pos, old_pos)
        if invalid_action:
            self.invalid_action_count = getattr(self, "invalid_action_count", 0) + 1
            if getattr(self, "reward_profile", "boundary_coverage") == "novelty_search":
                reward -= self.invalid_action_penalty
                r_breakdown["r_invalid"] -= self.invalid_action_penalty
                r_breakdown["r_penalty"] -= self.invalid_action_penalty

        recent_window = self._recent_cells[-self.pre_boundary_repeat_window :]
        if len(self.discovered_boundary) == 0 and cell in recent_window:
            reward -= self.pre_boundary_repeat_penalty
            r_breakdown["r_penalty"] -= self.pre_boundary_repeat_penalty

        if cell not in self.visited_cells:
            explore_reward = 0.01 if self.curriculum_stage == 1 else 0.02
            if self.curriculum_stage == 1:
                remaining = max(0.0, self.stage1_explore_reward_cap - self._episode_explore_reward_total)
                explore_reward = min(explore_reward, remaining)
            if explore_reward > 0.0:
                reward += explore_reward
                r_breakdown["r_explore"] += explore_reward
                self._episode_explore_reward_total += explore_reward

        if int(action) == 4:
            idle_penalty = -0.05 if self.curriculum_stage == 1 else -0.10
            reward += idle_penalty
            r_breakdown["r_penalty"] += idle_penalty

        peers = peer_new_positions if peer_new_positions is not None else self.drone_positions
        if len(peers) > 1:
            for j, other_pos in enumerate(peers):
                if j == drone_id:
                    continue
                if np.linalg.norm(new_pos - other_pos) < self.vision_radius * 0.8:
                    reward -= 0.15
                    r_breakdown["r_penalty"] -= 0.15
                    break

        # --- pre-boundary 搜索引导奖励：基于热势增量的弱引导 ---
        if len(self.discovered_boundary) == 0:
            potential_now = max(0.0, self.env_data.get_thermal_value(y, x))
            oy, ox = int(old_pos[0]), int(old_pos[1])
            potential_prev = max(0.0, self.env_data.get_thermal_value(oy, ox))
            delta = potential_now - potential_prev
            if delta > 0.0:
                coefficient = 1.0 if self.curriculum_stage == 1 else 2.0
                cap = 0.10 if self.curriculum_stage == 1 else 0.15
                r_search = min(coefficient * delta, cap)
                reward += r_search
                r_breakdown["r_search"] += r_search

        return float(reward), r_breakdown

    def _effective_post_target_step_penalty(self) -> float:
        first_target_step = getattr(self, "first_target_step", None)
        extra_steps = getattr(self, "post_target_extra_steps", None)
        if first_target_step is None:
            remaining_horizon = extra_steps or getattr(self, "max_steps", 600)
        elif extra_steps is not None:
            remaining_horizon = extra_steps
        else:
            remaining_horizon = max(
                1, getattr(self, "max_steps", 600) - first_target_step
            )
        milestone_scale = max(20.0, getattr(self, "post_target_milestone_70", 5.0))
        budget_cap = (
            getattr(self, "post_target_step_cost_fraction", 0.25) * milestone_scale
            / max(float(remaining_horizon), 1.0)
        )
        return -min(abs(self.post_target_step_penalty), budget_cap, 0.005)

    def _compute_profile_reward(
        self,
        pos: np.ndarray,
        cell_was_visited: bool,
        new_area_cells: int,
        severity_mean: float,
        severity_max: float,
        pre_boundary: bool,
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

        elif self.reward_profile == "novelty_search" and pre_boundary:
            strip_area = max(float(2 * self.vision_radius + 1), 1.0)
            novelty_fraction = float(np.clip(new_area_cells / strip_area, 0.0, 1.0))
            novelty_reward = self.novelty_reward_weight * novelty_fraction
            overlap_penalty = self.team_overlap_penalty * (1.0 - novelty_fraction)
            reward += novelty_reward - overlap_penalty
            r_breakdown["r_novelty"] += novelty_reward
            r_breakdown["r_overlap"] -= overlap_penalty
            r_breakdown["r_penalty"] -= overlap_penalty
            if new_area_cells == 0:
                reward -= self.novelty_revisit_penalty
                r_breakdown["r_revisit"] -= self.novelty_revisit_penalty
                r_breakdown["r_penalty"] -= self.novelty_revisit_penalty

        return float(reward), r_breakdown

    def _update_discovered_boundary(
        self, pos: np.ndarray, drone_idx: Optional[int] = None
    ) -> Tuple[int, int]:
        new_area_cells = self._mark_visible_region(pos)
        visible_boundary = self._get_visible_boundary_points(pos)
        if drone_idx is not None:
            self._mark_boundary_seen(
                int(drone_idx), visible_boundary, self.step_count + 1
            )

        new_points = 0
        for bp in visible_boundary:
            if bp not in self.discovered_boundary:
                new_points += 1
            self.discovered_boundary.add(bp)
            self.confirmed_boundary_mask[bp[0], bp[1]] = True

        return new_points, new_area_cells

    def _check_boundary_in_vision(self, pos: np.ndarray) -> bool:
        return len(self._get_visible_boundary_points(pos)) > 0

    def _update_contact_curriculum_state(self) -> Tuple[bool, bool]:
        if (
            self.curriculum_stage not in {1, 2}
            or self.first_boundary_step < 0
            or not hasattr(self, "boundary_points")
        ):
            return False, False
        boundary_visible = any(
            self._check_boundary_in_vision(pos) for pos in self.drone_positions
        )
        reacquired_now = False
        if self.curriculum_stage == 1:
            self.stage1_tracking_window.append(bool(boundary_visible))
            if len(self.stage1_tracking_window) > self.STAGE1_TRACK_WINDOW:
                self.stage1_tracking_window.pop(0)
            if self.stage1_contact_boundary_cells == 0:
                self.stage1_contact_boundary_cells = len(self.discovered_boundary)
            tracking_progress = max(
                0, len(self.discovered_boundary) - self.stage1_contact_boundary_cells
            )
            self.stage1_tracking_success = bool(
                len(self.stage1_tracking_window) >= self.STAGE1_TRACK_WINDOW
                and sum(self.stage1_tracking_window) >= self.STAGE1_TRACK_REQUIRED
                and len(self.discovered_boundary) >= self.STAGE1_MIN_UNIQUE_BOUNDARY
                and tracking_progress >= self.STAGE1_MIN_POST_CONTACT_PROGRESS
            )
        elif self.curriculum_stage == 2:
            if not boundary_visible:
                self.stage2_lost_after_contact = True
            elif self.stage2_lost_after_contact and not self.stage2_reacquire_success:
                self.stage2_reacquire_success = True
                reacquired_now = True
            self.stage2_tracking_window.append(bool(boundary_visible))
            if len(self.stage2_tracking_window) > self.STAGE2_TRACK_WINDOW:
                self.stage2_tracking_window.pop(0)
            stable_contact = bool(
                len(self.stage2_tracking_window) >= self.STAGE2_TRACK_WINDOW
                and sum(self.stage2_tracking_window) >= self.STAGE2_TRACK_REQUIRED
            )
            useful_reacquire = bool(
                self.stage2_reacquire_success
                and sum(self.stage2_tracking_window) >= self.STAGE2_REACQUIRE_MIN_VISIBLE
            )
            self.stage2_contact_success = stable_contact or useful_reacquire
        return bool(boundary_visible), reacquired_now

    def _check_done(self) -> Tuple[bool, str]:
        coverage = self._objective_coverage_rate()
        if self.termination_mode == "target_stop" and self._stage_target_reached(coverage):
            return True, "mission_complete"

        if (
            self.termination_mode == "post_target_train"
            and any(b <= 0 for b in self.drone_batteries)
        ):
            return True, "battery_depleted"
        if (
            self.termination_mode == "post_target_train"
            and self.post_target_extra_steps is not None
            and self.target_reached
            and self.step_count >= self.first_target_step + self.post_target_extra_steps
        ):
            return True, "curriculum_truncated"
        if self.step_count >= self.max_steps:
            if self.termination_mode in {"full_horizon", "post_target_train"}:
                return True, "horizon_reached"
            return True, "max_steps_reached"
        if any(b <= 0 for b in self.drone_batteries):
            return True, "battery_depleted"
        return False, "ongoing"

    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        rewards = []
        step_reward_breakdown = {k: 0.0 for k in self.episode_reward_breakdown}
        prev_on_curve = self._discovered_on_current_boundary_count()
        persistent_boundary = (
            getattr(self, "reward_profile", "boundary_coverage")
            == "persistent_boundary"
        )
        coordinator = getattr(self, "task_coordinator", None)
        fresh_before_action = 0.0
        self.coverage_action_gain = 0.0
        self.coverage_refresh_drop = 0.0
        if persistent_boundary:
            _, fresh_before_action = self._boundary_freshness_metrics()
        discovered_before_action = set(self.discovered_boundary)

        n_act = len(actions)
        old_positions = [self.drone_positions[i].copy() for i in range(n_act)]
        new_positions = [self._execute_action(old_positions[i], actions[i]) for i in range(n_act)]
        visible_boundary_sets = (
            [set(self._get_visible_boundary_points(pos)) for pos in new_positions]
            if persistent_boundary
            else [set() for _ in new_positions]
        )

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
            if coordinator is not None:
                new_points, new_area_cells = self._update_discovered_boundary(
                    new_pos, drone_idx=i
                )
            else:
                new_points, new_area_cells = self._update_discovered_boundary(new_pos)

            if (
                self.observation_profile
                in {"cooperative_exploration", "persistent_cooperative"}
                or getattr(self, "communication_enabled", False)
            ):
                self._mark_agent_visible_region(i, new_pos)

            if pre_boundary:
                self.pre_boundary_agent_steps = getattr(self, "pre_boundary_agent_steps", 0) + 1
                strip_area = max(float(2 * self.vision_radius + 1), 1.0)
                novelty_fraction = float(
                    np.clip(new_area_cells / strip_area, 0.0, 1.0)
                )
                self.team_overlap_sum = getattr(self, "team_overlap_sum", 0.0) + 1.0 - novelty_fraction
                if new_area_cells == 0:
                    self.pre_boundary_revisit_steps = getattr(self, "pre_boundary_revisit_steps", 0) + 1

            if (
                pre_boundary
                and new_points == 0
                and new_area_cells > 0
                and self.reward_profile != "novelty_search"
            ):
                r_area = self._pre_boundary_area_reward(new_area_cells)
                reward += r_area
                step_reward_breakdown["r_area_gain"] += r_area

            first_contact = new_points > 0 and self.first_boundary_step < 0
            if first_contact:
                self.first_boundary_step = self.step_count + 1
                contact_bonus = (
                    2.0
                    if self.curriculum_stage == 1
                    else 3.0
                    if self.curriculum_stage == 2
                    else 1.0
                )
                reward += contact_bonus
                step_reward_breakdown["r_contact"] += contact_bonus

            if new_points > 0:
                r_cov_gain = self._boundary_coverage_gain_reward(new_points)
                reward += r_cov_gain
                step_reward_breakdown["r_coverage_gain"] += r_cov_gain
                step_reward_breakdown["r_boundary"] += r_cov_gain

            profile_reward, profile_breakdown = self._compute_profile_reward(
                new_pos,
                cell_was_visited=cell_was_visited,
                new_area_cells=new_area_cells,
                severity_mean=severity_mean,
                severity_max=severity_max,
                pre_boundary=pre_boundary,
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

        if (
            self.observation_profile
            in {"cooperative_exploration", "persistent_cooperative"}
            or getattr(self, "communication_enabled", False)
        ):
            self._sync_exploration_knowledge(count_metrics=True)

        if coordinator is not None:
            coordinator.sync_connected_agents(self.step_count + 1)
            if coordinator.pending_role_switch_count:
                switch_cost = (
                    self.role_switch_penalty
                    * coordinator.pending_role_switch_count
                )
                rewards = [
                    reward - switch_cost / max(self.num_drones, 1)
                    for reward in rewards
                ]
                step_reward_breakdown["r_role_switch"] -= switch_cost
                coordinator.pending_role_switch_count = 0

        if persistent_boundary:
            _, fresh_after_action = self._boundary_freshness_metrics()
            self.coverage_action_gain = float(
                fresh_after_action - fresh_before_action
            )
            fresh_weight = {
                1: 5.0,
                2: 3.0,
            }.get(int(self.curriculum_stage), self.fresh_coverage_gain_weight)
            team_fresh_reward = (
                fresh_weight * self.coverage_action_gain
            )
            rewards = [
                reward + team_fresh_reward / max(self.num_drones, 1)
                for reward in rewards
            ]
            step_reward_breakdown["r_fresh_gain"] += team_fresh_reward
            for drone_idx in range(n_act):
                if self.curriculum_stage < 3:
                    continue
                peer_visible = set().union(
                    *[
                        visible_boundary_sets[peer]
                        for peer in range(n_act)
                        if peer != drone_idx
                    ]
                )
                unique_points = (
                    visible_boundary_sets[drone_idx]
                    - discovered_before_action
                    - peer_visible
                )
                assigned_gain = (
                    self.assigned_boundary_gain_weight
                    * len(unique_points)
                    / max(float(self.total_boundary_points), 1.0)
                )
                rewards[drone_idx] += assigned_gain
                step_reward_breakdown["r_assigned_gain"] += assigned_gain

        self.step_count += 1
        new_on_curve = self._discovered_on_current_boundary_count()
        self._coverage_gradient = (new_on_curve - prev_on_curve) / max(self.total_boundary_points, 1)

        self.boundary_refreshed = False
        self.coverage_before_boundary_refresh = self._boundary_coverage_rate()

        if self.step_count % 20 == 0:
            self.boundary_refreshed = True
            new_boundary = self.env_data.detect_fire_boundary(
                time_step=self.step_count,
                start_sim_time=self.env_data.training_start_sim_time,
            )
            self.env_data.boundary_points = list(new_boundary)
            self.boundary_points = list(new_boundary)
            self.total_boundary_points = max(len(self.boundary_points), 1)
            self._build_boundary_set()
            self._mark_current_boundary_ever()
            if self.boundary_points:
                self.fire_centroid = np.mean(
                    np.array(self.boundary_points, dtype=np.float32), axis=0
                )
            self.env_data._compute_thermal_field()
            self._refresh_confirmed_boundary_state()
            if coordinator is not None:
                coordinator.request_role_decision("boundary_refreshed")

        self.coverage_after_boundary_refresh = self._boundary_coverage_rate()
        if persistent_boundary:
            tolerant_coverage, fresh_coverage = self._boundary_freshness_metrics()
            self.coverage_refresh_drop = float(
                fresh_coverage - (fresh_before_action + self.coverage_action_gain)
            )
            self.tolerant_coverage_history.append(tolerant_coverage)
            self.fresh_coverage_history.append(fresh_coverage)
        else:
            tolerant_coverage = self.coverage_after_boundary_refresh
            fresh_coverage = self.coverage_after_boundary_refresh
            if self.boundary_refreshed:
                self.coverage_refresh_drop = float(
                    self.coverage_after_boundary_refresh
                    - self.coverage_before_boundary_refresh
                )
        if coordinator is not None:
            coordinator.maybe_request_periodic_decision(self.step_count)
        boundary_visible, reacquired_now = self._update_contact_curriculum_state()
        if self.curriculum_stage == 1 and boundary_visible:
            track_reward = 0.20
            rewards = [
                reward + track_reward / max(self.num_drones, 1)
                for reward in rewards
            ]
            step_reward_breakdown["r_track"] += track_reward
        elif self.curriculum_stage == 2 and boundary_visible:
            track_reward = 0.08
            rewards = [
                reward + track_reward / max(self.num_drones, 1)
                for reward in rewards
            ]
            step_reward_breakdown["r_track"] += track_reward
        if reacquired_now:
            reacquire_bonus = 1.5
            rewards = [
                reward + reacquire_bonus / max(self.num_drones, 1)
                for reward in rewards
            ]
            step_reward_breakdown["r_reacquire"] += reacquire_bonus
        objective_coverage = (
            fresh_coverage
            if persistent_boundary
            else self.coverage_after_boundary_refresh
        )
        if not self.target_reached and self._stage_target_reached(objective_coverage):
            self.target_reached = True
            self.first_target_step = self.step_count

        coverage = self.coverage_after_boundary_refresh
        self.coverage_history.append(float(objective_coverage))
        if self.boundary_refreshed and self.coverage_refresh_drop <= -0.05:
            self.major_refresh_count = getattr(self, "major_refresh_count", 0) + 1
            self._pending_refresh_step = self.step_count
            self._pending_recovery_target = min(
                0.60, max(0.0, self.coverage_before_boundary_refresh - 0.02)
            )
        if getattr(self, "_pending_refresh_step", -1) >= 0:
            elapsed = self.step_count - self._pending_refresh_step
            if objective_coverage >= self._pending_recovery_target:
                self.refresh_recovery_successes = (
                    getattr(self, "refresh_recovery_successes", 0) + 1
                )
                self.refresh_recovery_times = getattr(
                    self, "refresh_recovery_times", []
                )
                self.refresh_recovery_times.append(elapsed)
                self._pending_refresh_step = -1
            elif elapsed >= 100:
                self._pending_refresh_step = -1
        if self.termination_mode == "post_target_train":
            milestone_reward = 0.0
            if self.target_reached and not self.post_target_milestones["0.60"]:
                efficiency = 1.0 - np.clip(
                    self.first_target_step / max(float(self.max_steps), 1.0), 0.0, 1.0
                )
                milestone_reward += 20.0 + 10.0 * efficiency
                self.post_target_milestones["0.60"] = True
            if objective_coverage >= 0.70 and not self.post_target_milestones["0.70"]:
                milestone_reward += self.post_target_milestone_70
                self.post_target_milestones["0.70"] = True
            if objective_coverage >= 0.80 and not self.post_target_milestones["0.80"]:
                milestone_reward += self.post_target_milestone_80
                self.post_target_milestones["0.80"] = True
            if milestone_reward:
                rewards = [r + milestone_reward / self.num_drones for r in rewards]
                step_reward_breakdown["r_milestone"] += milestone_reward

            if self.target_reached:
                hold_reward = self.post_target_hold_weight * float(
                    np.clip(
                            (objective_coverage - self.post_target_goal)
                            / (1.0 - self.post_target_goal),
                        -1.0,
                        1.0,
                    )
                )
                rewards = [r + hold_reward / self.num_drones for r in rewards]
                step_reward_breakdown["r_hold"] += hold_reward

        done, done_reason = self._check_done()
        truncated = done_reason == "curriculum_truncated"
        terminated = bool(done and not truncated)
        timeout = done_reason == "max_steps_reached"
        zero_coverage_timeout = timeout and objective_coverage <= 1e-9
        zero_discovery_timeout = bool(
            done
            and done_reason in {"max_steps_reached", "horizon_reached"}
            and self.first_boundary_step < 0
        )

        if done:
            if done_reason == "mission_complete":
                efficiency = 1.0 - np.clip(self.step_count / max(float(self.max_steps), 1.0), 0.0, 1.0)
                if self.curriculum_stage == 1:
                    terminal_bonus = 3.0 + efficiency
                elif self.curriculum_stage == 2:
                    terminal_bonus = 4.0 + efficiency
                else:
                    terminal_bonus = 5.0 + 2.0 * efficiency
                rewards = [r + terminal_bonus / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] += terminal_bonus
            elif done_reason == "max_steps_reached":
                terminal_penalty = self._timeout_terminal_penalty(objective_coverage)
                rewards = [r - terminal_penalty / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] -= terminal_penalty
            elif done_reason == "battery_depleted":
                terminal_penalty = 5.0
                rewards = [r - terminal_penalty / self.num_drones for r in rewards]
                step_reward_breakdown["r_terminal"] -= terminal_penalty

            if (
                self.termination_mode == "post_target_train"
                and done_reason in {"horizon_reached", "curriculum_truncated"}
            ):
                tail100 = float(np.mean(self.coverage_history[-100:]))
                if self.target_reached:
                    tail_reward = self.post_target_tail_weight * float(
                        np.clip(
                            (tail100 - self.post_target_goal) / (1.0 - self.post_target_goal),
                            -1.0,
                            1.0,
                        )
                    )
                    rewards = [r + tail_reward / self.num_drones for r in rewards]
                    step_reward_breakdown["r_tail"] += tail_reward
                else:
                    terminal_penalty = self._timeout_terminal_penalty(tail100)
                    rewards = [r - terminal_penalty / self.num_drones for r in rewards]
                    step_reward_breakdown["r_terminal"] -= terminal_penalty

        if done_reason == "mission_complete" and self.first_boundary_step < 0:
            raise RuntimeError("mission completed without detecting a boundary")
        if (
            persistent_boundary
            and objective_coverage > 0.0
            and not np.any(self.boundary_last_seen_step >= 0)
        ):
            raise RuntimeError("positive fresh coverage without an episode-local boundary report")

        for key in self.episode_reward_breakdown:
            self.episode_reward_breakdown[key] += step_reward_breakdown[key]

        info = {
            "step": self.step_count,
            "boundary_coverage": coverage,
            "tolerant_boundary_coverage": tolerant_coverage,
            "fresh_boundary_coverage": fresh_coverage,
            "coverage_action_gain": float(self.coverage_action_gain),
            "coverage_refresh_drop": float(self.coverage_refresh_drop),
            "observable_progress": float(objective_coverage),
            "objective_coverage": float(objective_coverage),
            "avg_distance_to_fire": float(
                np.mean([np.linalg.norm(pos - self.fire_centroid) for pos in self.drone_positions])
            ),
            "done_reason": done_reason,
            "terminated": terminated,
            "truncated": truncated,
            "termination_mode": self.termination_mode,
            "target_reached": bool(self.target_reached),
            "first_target_step": int(self.first_target_step),
            "post_target_goal": float(self.post_target_goal),
            "post_target_tail100": float(np.mean(self.coverage_history[-100:])),
            "post_target_milestones": dict(self.post_target_milestones),
            "boundary_refreshed": bool(self.boundary_refreshed),
            "coverage_before_boundary_refresh": float(self.coverage_before_boundary_refresh),
            "coverage_after_boundary_refresh": float(self.coverage_after_boundary_refresh),
            "historical_boundary_union_coverage": self._historical_boundary_union_coverage_rate(),
            "scene_id": self.scene_id,
            "scene_key": self.scene_key,
            "observation_profile": self.observation_profile,
            "reward_profile": self.reward_profile,
            "vision_radius": self.vision_radius,
            "sensor_radius_cells": self.env_data.sensor_radius_cells,
            "max_steps": self.max_steps,
            "first_heat_step": int(self.first_heat_step),
            "first_boundary_step": int(self.first_boundary_step),
            "stage1_tracking_success": bool(
                getattr(self, "stage1_tracking_success", False)
            ),
            "stage1_tracking_steps": int(
                sum(getattr(self, "stage1_tracking_window", []))
            ),
            "stage1_unique_boundary_cells": int(len(self.discovered_boundary)),
            "stage2_contact_success": bool(
                getattr(self, "stage2_contact_success", False)
            ),
            "stage2_reacquire_success": bool(
                getattr(self, "stage2_reacquire_success", False)
            ),
            "stage2_tracking_steps": int(
                sum(getattr(self, "stage2_tracking_window", []))
            ),
            "timeout": bool(timeout),
            "zero_coverage_timeout": bool(zero_coverage_timeout),
            "zero_discovery_timeout": zero_discovery_timeout,
            "spawn_modes": list(self.spawn_modes),
            "curriculum_substage": str(getattr(self, "curriculum_substage", "1")),
            "major_refresh_count": int(getattr(self, "major_refresh_count", 0)),
            "refresh_recovery_successes": int(
                getattr(self, "refresh_recovery_successes", 0)
            ),
            "mean_refresh_recovery_time": (
                float(np.mean(getattr(self, "refresh_recovery_times", [])))
                if getattr(self, "refresh_recovery_times", [])
                else None
            ),
            "communication_available_rate": float(
                getattr(self, "communication_agent_steps", 0)
                / max(self.step_count * self.num_drones, 1)
            ),
            "shared_new_cells": int(getattr(self, "shared_new_cells", 0)),
            "pre_boundary_revisit_ratio": float(
                getattr(self, "pre_boundary_revisit_steps", 0)
                / max(getattr(self, "pre_boundary_agent_steps", 0), 1)
            ),
            "team_overlap_ratio": float(
                getattr(self, "team_overlap_sum", 0.0)
                / max(getattr(self, "pre_boundary_agent_steps", 0), 1)
            ),
            "invalid_action_count": int(getattr(self, "invalid_action_count", 0)),
            "current_roles": (
                [
                    CooperativeTaskCoordinator.ROLE_NAMES[role]
                    for role in coordinator.current_roles
                ]
                if coordinator is not None
                else []
            ),
            "role_switch_count": int(
                coordinator.role_switch_count
                if coordinator is not None
                else 0
            ),
            "invalid_role_count": int(
                coordinator.invalid_role_count
                if coordinator is not None
                else 0
            ),
            "task_conflict_count": int(
                coordinator.task_conflict_count
                if coordinator is not None
                else 0
            ),
            "expired_message_count": int(
                coordinator.expired_message_count
                if coordinator is not None
                else 0
            ),
            "communication_message_expirations": int(
                coordinator.expired_message_count if coordinator is not None else 0
            ),
            "boundary_report_expirations": int(
                coordinator.expired_boundary_report_count
                if coordinator is not None
                else 0
            ),
            "role_decision_reason": (
                coordinator.role_decision_reason
                if coordinator is not None
                else "disabled"
            ),
            "reward_breakdown": self.episode_reward_breakdown.copy() if done else None,
            "stage_target": 0.0
            if self.curriculum_stage == 1
            else (self.stage_targets[2] if self.curriculum_stage == 2 else self.stage_targets[3]),
        }

        return self._get_observation(), rewards, done, info

    def apply_joint_role_assignment(self, roles: List[int]) -> Dict:
        if self.task_coordinator is None:
            raise RuntimeError("hierarchical role coordination is disabled")
        self.task_coordinator.refresh_candidates(self.step_count)
        self.task_coordinator.apply_joint_roles(roles, self.step_count)
        return self._get_observation()

    def heuristic_joint_role_assignment(self) -> List[int]:
        if self.task_coordinator is None:
            raise RuntimeError("hierarchical role coordination is disabled")
        coordinator = self.task_coordinator
        coordinator.refresh_candidates(self.step_count)
        best_roles = [coordinator.SEARCH, coordinator.SEARCH]
        best_score = -float("inf")
        fire_known = bool(coordinator.fire_ever_seen)
        for joint_action, valid in enumerate(coordinator.joint_role_mask):
            if not valid:
                continue
            roles = [
                joint_action // coordinator.NUM_ROLES,
                joint_action % coordinator.NUM_ROLES,
            ]
            score = sum(
                float(coordinator.candidates[idx, role, 3])
                for idx, role in enumerate(roles)
            )
            if fire_known:
                score += 1.0 * roles.count(coordinator.TRACK)
                score += 0.5 * roles.count(coordinator.REACQUIRE)
            else:
                score += 1.0 * roles.count(coordinator.SEARCH)
            if score > best_score:
                best_score = score
                best_roles = roles
        return best_roles

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = int(stage)
        print(f"课程阶段已切换为 {self.curriculum_stage}")

    def set_curriculum_substage(self, substage: str):
        self.curriculum_substage = str(substage)

    def set_post_target_goal(self, goal: float):
        self.post_target_goal = float(np.clip(float(goal), 0.20, 0.80))

    def set_post_target_extra_steps(self, extra_steps: Optional[int]):
        self.post_target_extra_steps = None if extra_steps is None else max(1, int(extra_steps))

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
            "communication_enabled": self.communication_enabled,
            "communication_radius_factor": self.communication_radius_factor,
            "communication_radius": self.communication_radius,
            "action_mask_enabled": self.action_mask_enabled,
            "hierarchical_roles_enabled": self.hierarchical_roles_enabled,
            "role_obs_dim": (
                CooperativeTaskCoordinator.ROLE_OBS_DIM
                if self.hierarchical_roles_enabled
                else 0
            ),
            "num_roles": CooperativeTaskCoordinator.NUM_ROLES,
            "peer_state_ttl": self.peer_state_ttl,
            "track_report_ttl": self.track_report_ttl,
            "reacquire_report_ttl": self.reacquire_report_ttl,
            "role_decision_interval": self.role_decision_interval,
            "role_min_dwell_steps": self.role_min_dwell_steps,
            "boundary_match_radius": self.boundary_match_radius,
            "boundary_freshness_tau": self.boundary_freshness_tau,
            "role_switch_penalty": self.role_switch_penalty,
            "mask_thermal_below_signal": self.mask_thermal_below_signal,
            "stage2_target": self.stage_targets[2],
            "stage3_target": self.stage_targets[3],
            "init_area_percent": self.init_area_percent,
            "stage3_near_prob": self.stage3_near_prob,
            "post_target_extra_steps": self.post_target_extra_steps,
            "post_target_step_cost_fraction": self.post_target_step_cost_fraction,
            "termination_mode": self.termination_mode,
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
        }


if __name__ == "__main__":
    env = FireSearchBaselineEnvironment()
    obs = env.reset()
    print([o.shape for o in obs["local_obs"]], obs["global_state"].shape)
    obs, rewards, done, info = env.step([0 for _ in range(env.num_drones)])
    print(rewards, done, info)
