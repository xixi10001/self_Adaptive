import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ctde_ppo_baseline_train import RolePPOAgent, normalize_training_config
from rl_environment_baseline import (
    CooperativeTaskCoordinator,
    FireSearchBaselineEnvironment,
)


def _coordinator_env():
    env = SimpleNamespace()
    env.grid_size = (8, 8)
    env.num_drones = 2
    env.vision_radius = 2
    env.drone_positions = [
        np.array([2, 2], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
    ]
    env.agent_known_masks = [
        np.zeros(env.grid_size, dtype=np.bool_) for _ in range(env.num_drones)
    ]
    env.communication_available = [False, False]
    env.communication_enabled = True
    env.communication_radius = 4.0
    env.drone_batteries = [100.0, 100.0]
    env.max_battery = 100.0
    env.search_direction_mode = "octants"
    env.search_target_overlap_weight = 1.0
    env.search_target_stickiness_weight = 0.10
    env.search_target_min_novelty = 0.05
    env.search_target_refresh_steps = 20
    env.search_target_distance_factor = 1.5
    env.search_stagnation_steps = 9
    env.search_sector_switch_steps_first = 20
    env.search_sector_switch_steps = 20
    env.search_sector_cooldown_steps = 80
    env.search_min_pair_angle_deg = 90.0
    env.search_heat_pair_angle_deg = 45.0
    env.search_min_target_distance_factor = 2.0
    env.search_fallback_distance_factor = 1.5
    env.search_tie_tolerance = 1e-6

    def circular_window(y, x):
        y0, y1 = max(0, y - 2), min(8, y + 3)
        x0, x1 = max(0, x - 2), min(8, x + 3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        return y0, y1, x0, x1, (yy - y) ** 2 + (xx - x) ** 2 <= 4

    env._get_circular_window = circular_window
    return env


class MessageTtlTest(unittest.TestCase):
    def test_future_message_and_report_are_hard_cleared(self):
        coordinator = CooperativeTaskCoordinator(_coordinator_env(), 4, 6, 8, 10, 2)
        coordinator.peer_messages[0] = {
            "peer": 1,
            "position": np.array([2, 3]),
            "role": coordinator.TRACK,
            "received_step": 6,
            "source_step": 6,
        }
        coordinator.boundary_last_seen[0][1, 1] = 6

        coordinator.refresh_candidates(5)

        self.assertIsNone(coordinator.peer_messages[0])
        self.assertEqual(int(coordinator.boundary_last_seen[0][1, 1]), -1)
        role_obs = coordinator.role_observations(5)[0]
        self.assertEqual(float(role_obs[21]), 0.0)
        self.assertEqual(float(role_obs[22]), 0.0)

    def test_expired_message_and_spatial_report_are_physically_cleared(self):
        coordinator = CooperativeTaskCoordinator(_coordinator_env(), 4, 6, 8, 10, 2)
        coordinator.peer_messages[0] = {
            "peer": 1,
            "position": np.array([2, 3]),
            "role": coordinator.TRACK,
            "received_step": 0,
            "source_step": 0,
        }
        coordinator.boundary_last_seen[0][1, 1] = 0

        coordinator.refresh_candidates(3)
        self.assertIsNotNone(coordinator.peer_messages[0])
        self.assertGreater(coordinator.role_observations(3)[0][21], 0.0)

        coordinator.refresh_candidates(4)
        self.assertIsNone(coordinator.peer_messages[0])
        role_obs = coordinator.role_observations(4)[0]
        self.assertEqual(float(role_obs[21]), 0.0)
        self.assertEqual(float(role_obs[22]), 0.0)

        coordinator.refresh_candidates(8)
        self.assertEqual(int(coordinator.boundary_last_seen[0][1, 1]), -1)
        self.assertGreaterEqual(coordinator.expired_boundary_report_count, 1)

    def test_cross_drone_conflict_mask_only_applies_when_connected(self):
        env = _coordinator_env()
        coordinator = CooperativeTaskCoordinator(env, 4, 6, 8, 10, 2)
        coordinator.candidates[:, :, 0] = 1.0
        coordinator.candidates[:, coordinator.TRACK, 1:3] = 0.0

        coordinator._refresh_joint_mask()
        track_track = coordinator.TRACK * coordinator.NUM_ROLES + coordinator.TRACK
        self.assertEqual(int(coordinator.joint_role_mask[track_track]), 1)

        env.communication_available = [True, True]
        coordinator._refresh_joint_mask()
        self.assertEqual(int(coordinator.joint_role_mask[track_track]), 0)

    def test_search_directions_are_eight_equal_length_octants(self):
        coordinator = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=7
        )
        directions, sectors = coordinator._search_directions()

        self.assertEqual(directions.shape, (8, 2))
        np.testing.assert_array_equal(np.sort(sectors), np.arange(8))
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1),
            np.ones(8),
            atol=1e-6,
        )

    def test_equal_search_scores_use_reproducible_private_rng(self):
        coordinator_a = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=17
        )
        coordinator_b = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=17
        )

        sequence_a = [
            coordinator_a._random_best_index([1.0] * 8) for _ in range(24)
        ]
        np.random.seed(999)
        np.random.random(10000)
        sequence_b = [
            coordinator_b._random_best_index([1.0] * 8) for _ in range(24)
        ]

        self.assertEqual(sequence_a, sequence_b)
        self.assertGreater(len(set(sequence_a)), 1)

    def test_coordinator_rng_state_round_trip_replays_choices(self):
        coordinator = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=23
        )
        state = coordinator.get_rng_state()
        expected = [
            coordinator._random_best_index([1.0] * 8) for _ in range(12)
        ]
        coordinator.set_rng_state(state)
        actual = [
            coordinator._random_best_index([1.0] * 8) for _ in range(12)
        ]
        self.assertEqual(actual, expected)

    def test_connected_search_targets_are_jointly_separated(self):
        env = _coordinator_env()
        env.communication_available = [True, True]
        coordinator = CooperativeTaskCoordinator(env, 4, 6, 8, 10, 2)

        coordinator.refresh_candidates(0)

        target0 = coordinator._absolute_candidate_target(0, coordinator.SEARCH)
        target1 = coordinator._absolute_candidate_target(1, coordinator.SEARCH)
        self.assertEqual(coordinator._search_target_overlap(target0, target1), 0.0)

    def test_search_target_stays_stable_between_refreshes(self):
        env = _coordinator_env()
        env.communication_available = [True, True]
        coordinator = CooperativeTaskCoordinator(env, 4, 6, 8, 10, 2)
        coordinator.refresh_candidates(0)
        first = [target.copy() for target in coordinator.search_targets]

        coordinator.refresh_candidates(1)

        for expected, actual in zip(first, coordinator.search_targets):
            np.testing.assert_allclose(actual, expected)

    def test_forced_sector_switch_uses_20_step_scan_phases(self):
        env = _coordinator_env()
        env.communication_available = [True, True]
        coordinator = CooperativeTaskCoordinator(
            env, 4, 6, 8, 10, 2, random_seed=31
        )
        coordinator.refresh_candidates(0)
        initial_sectors = list(coordinator.search_target_sectors)

        coordinator.last_observation_progress_steps = [19, 19]
        coordinator.refresh_candidates(19)
        self.assertEqual(coordinator.forced_sector_switch_counts, [0, 0])

        coordinator.refresh_candidates(20)
        self.assertEqual(coordinator.forced_sector_switch_counts, [1, 1])
        self.assertEqual(coordinator.team_scan_phase, 1)
        for drone_idx, old_sector in enumerate(initial_sectors):
            self.assertNotEqual(
                coordinator.search_target_sectors[drone_idx], old_sector
            )
            self.assertGreater(
                int(coordinator.sector_cooldown_until[drone_idx][old_sector]),
                20,
            )
            self.assertEqual(coordinator.search_target_steps[drone_idx], 20)

        coordinator.last_observation_progress_steps = [39, 39]
        coordinator.refresh_candidates(39)
        self.assertEqual(coordinator.forced_sector_switch_counts, [1, 1])
        coordinator.refresh_candidates(40)
        self.assertEqual(coordinator.forced_sector_switch_counts, [2, 2])
        self.assertEqual(coordinator.team_scan_phase, 2)

    def test_scan_phase_rotates_cardinal_then_diagonal_axes(self):
        coordinator = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=31
        )
        coordinator.scan_base_cardinal_sector = 0
        coordinator.team_scan_phase = 0
        self.assertEqual(coordinator._scan_phase_sector_pair(), (0, 4))
        coordinator.team_scan_phase = 1
        self.assertEqual(coordinator._scan_phase_sector_pair(), (2, 6))
        coordinator.team_scan_phase = 2
        self.assertEqual(coordinator._scan_phase_sector_pair(), (1, 5))
        coordinator.team_scan_phase = 3
        self.assertEqual(coordinator._scan_phase_sector_pair(), (3, 7))

    def test_episode_direction_order_offset_is_seeded_and_nonconstant(self):
        first = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=53
        )
        second = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=53
        )
        self.assertEqual(first.direction_order_offset, second.direction_order_offset)
        np.testing.assert_array_equal(
            first._search_directions()[1],
            second._search_directions()[1],
        )
        offsets = {
            CooperativeTaskCoordinator(
                _coordinator_env(), 4, 6, 8, 10, 2, random_seed=seed
            ).direction_order_offset
            for seed in range(16)
        }
        self.assertGreater(len(offsets), 1)

    def test_visible_heat_relaxes_connected_pair_angle_to_45_degrees(self):
        env = _coordinator_env()
        env.communication_available = [True, True]
        coordinator = CooperativeTaskCoordinator(
            env, 4, 6, 8, 10, 2, random_seed=37
        )
        candidate = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        options = [
            [(np.array([0.0, 3.0], dtype=np.float32), candidate, 1)],
            [(np.array([0.0, 7.0], dtype=np.float32), candidate, 2)],
        ]
        angle, distance = coordinator._pair_angle_distance(
            options[0][0][0], options[1][0][0]
        )
        self.assertGreaterEqual(angle, 45.0)
        self.assertLess(angle, 90.0)
        self.assertGreaterEqual(distance, 2.0 * env.vision_radius)

        coordinator.observe_heat_signal(0, True, 5)
        selected = coordinator._select_connected_search_pair(options, 5)

        self.assertEqual(len(selected), 2)
        self.assertEqual(coordinator.heat_separation_relaxation_count, 1)
        self.assertEqual(coordinator.edge_separation_fallback_count, 0)
        self.assertEqual(coordinator.connected_separation_violation_count, 0)

    def test_observed_boundary_disables_search_sector_timer(self):
        coordinator = CooperativeTaskCoordinator(
            _coordinator_env(), 4, 6, 8, 10, 2, random_seed=41
        )
        coordinator.refresh_candidates(0)
        coordinator.observe_boundary(0, [(1, 1)], 20)
        coordinator.refresh_candidates(120)
        self.assertEqual(coordinator.forced_sector_switch_counts[0], 0)

    def test_search_candidates_do_not_depend_on_hidden_fire_geometry(self):
        env_a = _coordinator_env()
        env_b = _coordinator_env()
        env_a.fire_centroid = np.array([0.0, 0.0])
        env_b.fire_centroid = np.array([7.0, 7.0])
        env_a.boundary_points = [(0, 0)]
        env_b.boundary_points = [(7, 7)]
        coordinator_a = CooperativeTaskCoordinator(env_a, 4, 6, 8, 10, 2)
        coordinator_b = CooperativeTaskCoordinator(env_b, 4, 6, 8, 10, 2)

        for step in [0, 20, 40]:
            coordinator_a.refresh_candidates(step)
            coordinator_b.refresh_candidates(step)
            np.testing.assert_allclose(
                coordinator_a.candidates[:, coordinator_a.SEARCH],
                coordinator_b.candidates[:, coordinator_b.SEARCH],
            )
            self.assertEqual(
                coordinator_a.search_target_sectors,
                coordinator_b.search_target_sectors,
            )

    def test_disconnected_local_event_locks_unaffected_peer_role(self):
        env = _coordinator_env()
        coordinator = CooperativeTaskCoordinator(env, 4, 6, 8, 10, 2)
        coordinator.candidates[:, :, 0] = 1.0
        coordinator.role_decision_required = False
        coordinator.role_decision_agents = [False, False]
        coordinator.request_role_decision("local_fire_detected", drone_idx=0)
        coordinator._refresh_joint_mask()

        valid_pairs = np.flatnonzero(coordinator.joint_role_mask)
        peer_roles = valid_pairs % coordinator.NUM_ROLES
        self.assertTrue(np.all(peer_roles == coordinator.current_roles[1]))


class RolePpoTest(unittest.TestCase):
    def make_agent(self):
        return RolePPOAgent(
            role_obs_dim=27,
            global_state_dim=19,
            num_roles=3,
            device=torch.device("cpu"),
            actor_lr=2e-4,
            critic_lr=5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            ppo_epochs=1,
            batch_size=8,
        )

    def test_joint_mask_is_applied_before_role_sampling(self):
        agent = self.make_agent()
        mask = np.zeros(9, dtype=np.int8)
        mask[5] = 1
        roles, _, action = agent.select_joint_roles(
            [np.zeros(27, dtype=np.float32) for _ in range(2)],
            mask,
            deterministic=True,
        )
        self.assertEqual(action, 5)
        self.assertEqual(roles, [1, 2])

    def test_smdp_truncation_bootstraps_gamma_to_option_duration(self):
        agent = self.make_agent()
        advantages, returns = agent.compute_smdp_gae(
            rewards=torch.tensor([0.0]),
            durations=torch.tensor([2.0]),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([True]),
            values=torch.tensor([1.0]),
            next_values=torch.tensor([2.0]),
        )
        expected = 0.99**2 * 2.0 - 1.0
        self.assertAlmostEqual(float(advantages[0]), expected, places=6)
        self.assertAlmostEqual(float(returns[0]), expected + 1.0, places=6)


class PersistentCoverageTest(unittest.TestCase):
    def test_tolerance_radius_survives_small_boundary_refresh_shift(self):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.boundary_points = [(3, 4)]
        env.boundary_last_seen_step = np.full((8, 8), -1, dtype=np.int32)
        env.boundary_last_seen_step[3, 3] = 10
        env.boundary_match_radius = 1
        env.boundary_freshness_tau = 40.0
        env.grid_size = (8, 8)
        env.step_count = 10

        tolerant, fresh = env._boundary_freshness_metrics()
        self.assertEqual(tolerant, 1.0)
        self.assertEqual(fresh, 1.0)

    def test_future_episode_timestamp_is_not_treated_as_fresh(self):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.boundary_points = [(3, 3)]
        env.boundary_last_seen_step = np.full((8, 8), -1, dtype=np.int32)
        env.boundary_last_seen_step[3, 3] = 50
        env.boundary_match_radius = 1
        env.boundary_freshness_tau = 40.0
        env.grid_size = (8, 8)
        env.step_count = 0

        self.assertEqual(env._boundary_freshness_metrics(), (0.0, 0.0))

    def test_persistent_reward_selects_matching_observation_profile(self):
        config = normalize_training_config({"reward_profile": "persistent_boundary"})
        self.assertEqual(config["observation_profile"], "persistent_cooperative")
        self.assertTrue(config["hierarchical_roles_enabled"])
        self.assertTrue(config["communication_enabled"])
        self.assertTrue(config["action_mask_enabled"])
        self.assertTrue(config["mask_thermal_below_signal"])
        self.assertEqual(config["search_direction_mode"], "octants")
        self.assertEqual(config["search_target_refresh_steps"], 20)
        self.assertEqual(config["search_target_distance_factor"], 1.5)
        self.assertEqual(config["search_sector_switch_steps_first"], 20)
        self.assertEqual(config["search_sector_switch_steps"], 20)
        self.assertEqual(config["search_min_pair_angle_deg"], 90.0)
        self.assertEqual(config["search_heat_pair_angle_deg"], 45.0)


if __name__ == "__main__":
    unittest.main()
