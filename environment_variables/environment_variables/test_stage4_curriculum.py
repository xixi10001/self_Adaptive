import unittest

import numpy as np
import torch
import torch.nn as nn

from ctde_ppo_baseline_train import (
    CTDE_PPO_Agent,
    CurriculumManager,
    Stage4StepMixScheduler,
)
from rl_environment_baseline import FireSearchBaselineEnvironment


class _FirstFeatureValue(nn.Module):
    def forward(self, states):
        return states[:, :1]


class Stage4GaeTest(unittest.TestCase):
    def make_agent(self):
        agent = CTDE_PPO_Agent(
            local_obs_dim=2,
            global_state_dim=1,
            action_dim=2,
            num_agents=1,
            gamma=0.99,
            gae_lambda=0.95,
            batch_size=32,
            device="cpu",
        )
        agent.critic = _FirstFeatureValue()
        return agent

    def test_curriculum_truncation_bootstraps_next_value_and_stops_trace(self):
        agent = self.make_agent()
        advantages, returns = agent.compute_gae(
            rewards_list=[[0.0]],
            terminated=[False],
            truncated=[True],
            global_states=np.array([[1.0]], dtype=np.float32),
            next_global_states=np.array([[2.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(advantages[0]), 0.98, places=6)
        self.assertAlmostEqual(float(returns[0]), 1.98, places=6)

    def test_true_termination_does_not_bootstrap(self):
        agent = self.make_agent()
        advantages, returns = agent.compute_gae(
            rewards_list=[[0.0]],
            terminated=[True],
            truncated=[False],
            global_states=np.array([[1.0]], dtype=np.float32),
            next_global_states=np.array([[2.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(advantages[0]), -1.0, places=6)
        self.assertAlmostEqual(float(returns[0]), 0.0, places=6)


class Stage4EnvironmentTest(unittest.TestCase):
    def make_env(self, step_count):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.termination_mode = "post_target_train"
        env.curriculum_stage = 3
        env.stage_targets = {2: 0.15, 3: 0.60}
        env.target_reached = True
        env.first_target_step = 10
        env.post_target_extra_steps = 100
        env.step_count = step_count
        env.max_steps = 600
        env.drone_batteries = [100.0]
        env._boundary_coverage_rate = lambda: 0.65
        return env

    def test_extra_horizon_is_curriculum_truncation(self):
        self.assertEqual(self.make_env(109)._check_done(), (False, "ongoing"))
        self.assertEqual(
            self.make_env(110)._check_done(),
            (True, "curriculum_truncated"),
        )


class Stage4SchedulerTest(unittest.TestCase):
    def test_ratio_ramps_over_ten_updates_and_step_debt_corrects_mode(self):
        scheduler = Stage4StepMixScheduler(start_update=0)
        scheduler.set_target_ratio(0.30, update_step=0)

        self.assertAlmostEqual(scheduler.current_target_ratio(0), 0.0)
        self.assertAlmostEqual(scheduler.current_target_ratio(5), 0.15)
        self.assertAlmostEqual(scheduler.current_target_ratio(10), 0.30)

        scheduler.record("rehearsal", 100)
        self.assertEqual(scheduler.choose_mode(10), "continuation")
        scheduler.record("continuation", 100, first_target_step=60)
        self.assertEqual(scheduler.choose_mode(10), "rehearsal")
        self.assertAlmostEqual(scheduler.stats(10)["realized_post_target_ratio"], 0.20)


class Stage4ProgressionTest(unittest.TestCase):
    def test_stage4_requires_two_protected_validation_passes(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3
        manager.stage_episodes[3] = 300
        manager._s3_target_idx = len(manager.STAGE3_TARGET_LADDER) - 1
        manager._s3_near_idx = len(manager.STAGE3_NEAR_LADDER) - 1
        self.assertTrue(
            manager.can_enter_stage4(
                {"success_rate": 0.60, "zero_coverage_timeout_rate": 0.10}
            )
        )

        baseline = {
            "boundary_found_rate": 0.95,
            "target_reach_rate": 0.63,
            "mean_tail100_coverage": 0.57,
            "mean_current_coverage_auc": 0.44,
            "mean_hold_ratio_by_threshold": {"0.60": 0.46},
        }
        manager.enter_stage4(baseline)
        passing = {
            "boundary_found_rate": 0.94,
            "target_reach_rate": 0.62,
            "mean_tail100_coverage": 0.58,
            "mean_current_coverage_auc": 0.45,
            "mean_hold_ratio_by_threshold": {"0.60": 0.47},
        }

        self.assertFalse(manager.update_stage4_validation(passing))
        self.assertTrue(manager.update_stage4_validation(passing))
        self.assertEqual(manager.stage4_level, "4B")


if __name__ == "__main__":
    unittest.main()
