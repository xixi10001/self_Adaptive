import unittest

from rl_environment_baseline import FireSearchBaselineEnvironment


class BoundaryRewardDesignTest(unittest.TestCase):
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

    def test_zero_coverage_timeout_has_extra_terminal_penalty(self):
        env = object.__new__(FireSearchBaselineEnvironment)
        env.curriculum_stage = 3
        env.stage_targets = {2: 0.15, 3: 0.60}

        zero_coverage_penalty = env._timeout_terminal_penalty(coverage=0.0)
        partial_coverage_penalty = env._timeout_terminal_penalty(coverage=0.30)

        self.assertGreater(zero_coverage_penalty, partial_coverage_penalty + 20.0)


if __name__ == "__main__":
    unittest.main()
