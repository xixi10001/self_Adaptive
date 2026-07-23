import unittest

import numpy as np

from rl_environment_baseline import FireSearchBaselineEnvironment


class _FakeEnvData:
    def __init__(self, heat: float, fire_count: int = 0):
        self.heat = heat
        self.fire_count = fire_count

    def get_thermal_value(self, row, col):
        if isinstance(self.heat, dict):
            return self.heat.get((int(row), int(col)), 0.0)
        return self.heat

    def get_local_fire_info(self, row, col, radius):
        return {"fire_count": self.fire_count}


class BoundaryRewardDesignTest(unittest.TestCase):
    def _cooperative_reward_env(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.grid_size = (10, 10)
        env.vision_radius = 1
        env.agent_observed_masks = [
            np.zeros(env.grid_size, dtype=np.bool_),
            np.zeros(env.grid_size, dtype=np.bool_),
        ]
        env.pre_boundary_team_novelty_weight = 0.20
        env.pre_boundary_unique_novelty_weight = 0.10
        env.pre_boundary_overlap_weight = 0.10
        env.pre_boundary_revisit_weight = 0.06
        env.pre_boundary_reward_episode_cap = 4.0
        env._episode_pre_boundary_reward_total = 0.0
        env.pre_boundary_agent_steps = 0
        env.pre_boundary_revisit_steps = 0.0
        env.team_overlap_sum = 0.0
        return env

    def test_stage1_short_tracking_window_matches_foundation_task(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.curriculum_stage = 1
        env.first_boundary_step = 1
        env.boundary_points = [(0, 0)]
        env.drone_positions = [np.array([0, 0])]
        env.discovered_boundary = {(0, i) for i in range(8)}
        env.stage1_discovered_within_deadline = True
        env.stage1_tracking_window = []
        env.stage1_tracking_success = False
        env._check_boundary_in_vision = lambda pos: True

        env.step_count = env.first_boundary_step
        env._update_contact_curriculum_state()
        self.assertEqual(env.stage1_tracking_window, [])

        for step in range(
            env.first_boundary_step + 1,
            env.first_boundary_step + env.STAGE1_TRACK_WINDOW + 1,
        ):
            env.step_count = step
            env._update_contact_curriculum_state()

        self.assertTrue(env.stage1_tracking_success)
        self.assertEqual(
            len(env.stage1_tracking_window),
            env.STAGE1_TRACK_WINDOW,
        )

    def test_boundary_coverage_reward_dominates_area_exploration_reward(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.total_boundary_points = 100
        env.coverage_gain_weight = 40.0
        env.coverage_gain_clip = 2.0
        env.pre_boundary_area_gain_weight = 0.35
        env.pre_boundary_area_gain_clip = 0.08
        env.vision_radius = 16

        coverage_reward = env._boundary_coverage_gain_reward(new_points=5)
        area_reward = env._pre_boundary_area_reward(new_area_cells=100)

        self.assertGreater(coverage_reward, area_reward * 10.0)

    def test_freshness_uses_explicit_transition_reference_step(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.boundary_points = [(1, 1)]
        env.grid_size = (3, 3)
        env.boundary_match_radius = 0
        env.boundary_freshness_tau = 10.0
        env.step_count = 4
        env.boundary_last_seen_step = np.full((3, 3), -1, dtype=np.int32)
        env.boundary_last_seen_step[1, 1] = 5

        self.assertEqual(env._boundary_freshness_metrics(), (0.0, 0.0))
        self.assertEqual(
            env._boundary_freshness_metrics(reference_step=5),
            (1.0, 1.0),
        )

    def test_stage1_area_shaping_has_episode_budget(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.curriculum_stage = 1
        env.vision_radius = 1
        env.pre_boundary_area_gain_weight = 10.0
        env.pre_boundary_area_gain_clip = 1.0
        env._episode_area_reward_total = 0.0

        rewards = [env._pre_boundary_area_reward(new_area_cells=9) for _ in range(3)]

        self.assertAlmostEqual(sum(rewards), env.STAGE1_AREA_REWARD_CAP)
        self.assertAlmostEqual(env._episode_area_reward_total, env.STAGE1_AREA_REWARD_CAP)

    def test_zero_coverage_timeout_penalty_is_extra_but_bounded(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.curriculum_stage = 3
        env.stage_targets = {2: 0.15, 3: 0.60}

        zero_coverage_penalty = env._timeout_terminal_penalty(coverage=0.0)
        partial_coverage_penalty = env._timeout_terminal_penalty(coverage=0.30)

        self.assertGreater(zero_coverage_penalty, partial_coverage_penalty)
        self.assertLessEqual(zero_coverage_penalty, 6.0)

    def test_heat_signal_uses_conservative_thermal_threshold(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.vision_radius = 16
        pos = np.array([10, 10])

        env.env_data = _FakeEnvData(heat=0.49, fire_count=0)
        below = env._get_heat_signal_features(pos)
        self.assertFalse(below["thermal_sensor_signal"])
        self.assertFalse(below["has_heat_signal"])

        env.env_data = _FakeEnvData(heat=0.50, fire_count=0)
        at_threshold = env._get_heat_signal_features(pos)
        self.assertTrue(at_threshold["thermal_sensor_signal"])
        self.assertTrue(at_threshold["has_heat_signal"])

        env.env_data = _FakeEnvData(heat=0.0, fire_count=1)
        local_fire = env._get_heat_signal_features(pos)
        self.assertFalse(local_fire["thermal_sensor_signal"])
        self.assertTrue(local_fire["local_fire_visible"])
        self.assertTrue(local_fire["has_heat_signal"])

    def test_cooperative_search_rewards_separation_over_overlapping_footprints(self):
        separated = self._cooperative_reward_env()
        separated_rewards, _ = separated._pre_boundary_cooperative_search_reward(
            [np.array([2, 2]), np.array([7, 7])]
        )
        overlapping = self._cooperative_reward_env()
        overlapping_rewards, breakdown = overlapping._pre_boundary_cooperative_search_reward(
            [np.array([5, 5]), np.array([5, 5])]
        )

        self.assertGreater(sum(separated_rewards), sum(overlapping_rewards))
        self.assertLess(breakdown["r_overlap"], 0.0)
        self.assertGreater(overlapping.team_overlap_sum, 0.0)

    def test_pre_boundary_positive_search_reward_has_episode_cap(self):
        env = self._cooperative_reward_env()
        env.pre_boundary_reward_episode_cap = 0.10

        for _ in range(3):
            env._pre_boundary_cooperative_search_reward(
                [np.array([2, 2]), np.array([7, 7])]
            )

        self.assertAlmostEqual(env._episode_pre_boundary_reward_total, 0.10)

    def test_hidden_subthreshold_heat_cannot_shape_search_reward(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.vision_radius = 16
        env.curriculum_stage = 2
        env.mask_thermal_below_signal = True
        env._episode_search_reward_total = 0.0
        env.env_data = _FakeEnvData({(1, 1): 0.10, (1, 2): 0.49}, fire_count=0)

        hidden_reward = env._pre_boundary_heat_progress_reward(
            np.array([1, 1]), np.array([1, 2])
        )
        env.env_data = _FakeEnvData({(1, 1): 0.10, (1, 2): 0.60}, fire_count=0)
        visible_reward = env._pre_boundary_heat_progress_reward(
            np.array([1, 1]), np.array([1, 2])
        )

        self.assertEqual(hidden_reward, 0.0)
        self.assertGreater(visible_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
