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

    def circular_window(y, x):
        y0, y1 = max(0, y - 2), min(8, y + 3)
        x0, x1 = max(0, x - 2), min(8, x + 3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        return y0, y1, x0, x1, (yy - y) ** 2 + (xx - x) ** 2 <= 4

    env._get_circular_window = circular_window
    return env


class MessageTtlTest(unittest.TestCase):
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

    def test_equal_search_scores_use_agent_index_only_as_tie_breaker(self):
        coordinator = CooperativeTaskCoordinator(_coordinator_env(), 4, 6, 8, 10, 2)
        first = coordinator._candidate(0, coordinator.SEARCH, 0)
        second = coordinator._candidate(1, coordinator.SEARCH, 0)
        self.assertFalse(np.allclose(first[1:3], second[1:3]))

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

    def test_persistent_reward_selects_matching_observation_profile(self):
        config = normalize_training_config({"reward_profile": "persistent_boundary"})
        self.assertEqual(config["observation_profile"], "persistent_cooperative")
        self.assertTrue(config["hierarchical_roles_enabled"])
        self.assertTrue(config["communication_enabled"])
        self.assertTrue(config["action_mask_enabled"])
        self.assertTrue(config["mask_thermal_below_signal"])


if __name__ == "__main__":
    unittest.main()
