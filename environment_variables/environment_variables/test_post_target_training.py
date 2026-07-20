import os
import tempfile
import unittest

import numpy as np

from ctde_ppo_baseline_train import (
    CTDE_PPO_Agent,
    _advance_post_target_goal,
    _post_target_optimizer_settings,
    normalize_training_config,
)
from rl_environment_baseline import FireSearchBaselineEnvironment


class _FakeEnvData:
    sensor_radius_cells = 16

    @staticmethod
    def get_wind_effect(row, col, movement):
        return {"battery_penalty": 0.0}


class PostTargetEnvironmentTest(unittest.TestCase):
    def make_step_env(
        self,
        mode="post_target_train",
        coverage=0.60,
        step_count=98,
        max_steps=600,
        target_reached=False,
    ):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.termination_mode = mode
        env.curriculum_stage = 3
        env.stage_targets = {2: 0.15, 3: 0.60}
        env.num_drones = 1
        env.step_count = step_count
        env.max_steps = max_steps
        env.drone_positions = [np.array([0, 0])]
        env.drone_batteries = [1000.0]
        env.drone_momentums = [np.zeros(2, dtype=np.float32)]
        env.discovered_boundary = {(0, i) for i in range(int(round(coverage * 10)))}
        env._boundary_set = {(0, i) for i in range(10)}
        env.total_boundary_points = 10
        env.visited_cells = set()
        env._recent_cells = []
        env.pre_boundary_repeat_window = 12
        env.first_heat_step = -1
        env.first_boundary_step = 1
        env.target_reached = target_reached
        env.first_target_step = 10 if target_reached else -1
        env.boundary_refreshed = False
        env.coverage_before_boundary_refresh = coverage
        env.coverage_after_boundary_refresh = coverage
        env.coverage_history = []
        env.post_target_goal = 0.60
        env.post_target_step_penalty = -0.02
        env.post_target_hold_weight = 0.10
        env.post_target_tail_weight = 20.0
        env.post_target_milestone_70 = 5.0
        env.post_target_milestone_80 = 10.0
        env.post_target_extra_steps = None
        env.post_target_milestones = {"0.60": False, "0.70": False, "0.80": False}
        env.episode_reward_breakdown = env._empty_reward_breakdown()
        env._coverage_gradient = 0.0
        env.boundary_ever_mask = np.ones((1, 10), dtype=np.bool_)
        env.confirmed_boundary_mask = np.zeros((1, 10), dtype=np.bool_)
        env.confirmed_boundary_mask[0, : len(env.discovered_boundary)] = True
        env.fire_centroid = np.array([0.0, 0.0])
        env.scene_id = 1
        env.scene_key = "scene"
        env.observation_profile = "baseline"
        env.reward_profile = "boundary_coverage"
        env.vision_radius = 16
        env.spawn_modes = ["far"]
        env.env_data = _FakeEnvData()
        env._execute_action = lambda old_pos, action: old_pos.copy()
        env._compute_reward = lambda *args, **kwargs: (0.0, env._empty_reward_breakdown())
        env._compute_profile_reward = lambda *args, **kwargs: (
            0.0,
            env._empty_reward_breakdown(),
        )
        env._get_heat_signal_features = lambda pos: {"has_heat_signal": False}
        env._update_discovered_boundary = lambda pos: (0, 0)
        env._get_observation = lambda: {"local_obs": [], "global_state": np.zeros(1)}
        return env

    def test_target_stop_keeps_old_terminal_bonus(self):
        env = self.make_step_env(mode="target_stop")

        _, _, done, info = env.step([4])

        expected = 20.0 + 10.0 * (1.0 - 99.0 / 600.0)
        self.assertTrue(done)
        self.assertEqual(info["done_reason"], "mission_complete")
        self.assertAlmostEqual(info["reward_breakdown"]["r_terminal"], expected)
        self.assertEqual(info["reward_breakdown"]["r_milestone"], 0.0)

    def test_full_horizon_has_no_post_target_training_rewards(self):
        env = self.make_step_env(mode="full_horizon", coverage=0.70, step_count=0, max_steps=1)

        _, _, done, info = env.step([4])

        self.assertTrue(done)
        self.assertEqual(info["done_reason"], "horizon_reached")
        self.assertEqual(info["reward_breakdown"]["r_milestone"], 0.0)
        self.assertEqual(info["reward_breakdown"]["r_hold"], 0.0)
        self.assertEqual(info["reward_breakdown"]["r_tail"], 0.0)

    def test_post_target_milestones_are_awarded_once(self):
        env = self.make_step_env()

        env.step([4])
        first_reward = env.episode_reward_breakdown["r_milestone"]
        env.step_count = 100
        env.discovered_boundary = {(0, i) for i in range(8)}
        env.step([4])
        all_milestones = env.episode_reward_breakdown["r_milestone"]
        env.discovered_boundary = {(0, i) for i in range(5)}
        env.step([4])
        env.discovered_boundary = {(0, i) for i in range(8)}
        env.step([4])

        expected_first = 20.0 + 10.0 * (1.0 - 99.0 / 600.0)
        self.assertAlmostEqual(first_reward, expected_first)
        self.assertAlmostEqual(all_milestones, expected_first + 5.0 + 10.0)
        self.assertAlmostEqual(env.episode_reward_breakdown["r_milestone"], all_milestones)

    def test_horizon_tail_reward_uses_tail100_mean(self):
        env = self.make_step_env(
            coverage=0.0,
            step_count=0,
            max_steps=1,
            target_reached=True,
        )
        env.post_target_milestones = {"0.60": True, "0.70": True, "0.80": False}
        env.coverage_history = [0.70] * 99

        _, _, _, info = env.step([4])

        expected_tail_mean = 0.693
        expected_reward = 20.0 * (expected_tail_mean - 0.60) / 0.40
        self.assertAlmostEqual(info["post_target_tail100"], expected_tail_mean)
        self.assertAlmostEqual(info["reward_breakdown"]["r_tail"], expected_reward)

    def test_never_reaching_target_is_not_double_penalized(self):
        env = self.make_step_env(coverage=0.30, step_count=0, max_steps=1)

        _, _, _, info = env.step([4])

        self.assertLess(info["reward_breakdown"]["r_terminal"], 0.0)
        self.assertEqual(info["reward_breakdown"]["r_tail"], 0.0)

    def test_battery_depletion_at_horizon_cannot_receive_tail_reward(self):
        env = self.make_step_env(
            coverage=0.70,
            step_count=0,
            max_steps=1,
            target_reached=True,
        )
        env.drone_batteries = [0.0]
        env.post_target_milestones = {"0.60": True, "0.70": True, "0.80": False}

        _, _, _, info = env.step([4])

        self.assertEqual(info["done_reason"], "battery_depleted")
        self.assertEqual(info["reward_breakdown"]["r_tail"], 0.0)
        self.assertEqual(info["reward_breakdown"]["r_terminal"], -5.0)

    def test_post_target_step_penalty_switches_after_target(self):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.curriculum_stage = 3
        env.termination_mode = "post_target_train"
        env.post_target_step_penalty = -0.02
        env._boundary_set = set()
        env.discovered_boundary = {(9, 9)}
        env._recent_cells = []
        env.pre_boundary_repeat_window = 12
        env.visited_cells = {(0, 0)}
        env.vision_radius = 16
        env.drone_positions = [np.array([0, 0])]
        env._episode_explore_reward_total = 0.0

        env.target_reached = False
        before, _ = env._compute_reward(0, np.array([0, 0]), np.array([0, 0]), 0)
        env.target_reached = True
        after, _ = env._compute_reward(0, np.array([0, 0]), np.array([0, 0]), 0)

        self.assertAlmostEqual(after - before, 0.06)


class PostTargetTrainingControlTest(unittest.TestCase):
    def passing_summary(self):
        return {
            "boundary_found_rate": 0.95,
            "mean_tail100_coverage": 0.63,
            "mean_hold_ratio_by_threshold": {"0.60": 0.56},
            "threshold_reach_rate": {"0.70": 0.0},
        }

    def test_goal_requires_two_consecutive_validation_passes(self):
        ladder = [0.60, 0.65, 0.70]
        index, count, advanced = _advance_post_target_goal(
            0, 0, self.passing_summary(), ladder, 2
        )
        self.assertEqual((index, count, advanced), (0, 1, False))

        index, count, advanced = _advance_post_target_goal(
            index, count, self.passing_summary(), ladder, 2
        )
        self.assertEqual((index, count, advanced), (1, 0, True))

        failed = self.passing_summary()
        failed["boundary_found_rate"] = 0.94
        self.assertEqual(
            _advance_post_target_goal(index, 1, failed, ladder, 2),
            (1, 0, False),
        )

    def test_optimizer_warmup_matches_agreed_endpoints(self):
        config = normalize_training_config()

        first = _post_target_optimizer_settings(0, config)
        last_warmup = _post_target_optimizer_settings(14, config)
        normal = _post_target_optimizer_settings(15, config)

        self.assertAlmostEqual(first["actor_lr"], 5e-5)
        self.assertAlmostEqual(first["critic_lr"], 1.25e-4)
        self.assertEqual(first["clip_epsilon"], 0.15)
        self.assertEqual(first["max_grad_norm"], 0.25)
        self.assertAlmostEqual(last_warmup["actor_lr"], 1e-4)
        self.assertTrue(last_warmup["override_actor_lr"])
        self.assertAlmostEqual(normal["critic_lr"], 2.5e-4)
        self.assertEqual(normal["clip_epsilon"], 0.2)
        self.assertEqual(normal["max_grad_norm"], 0.5)
        self.assertTrue(normal["override_actor_lr"])

        kl_config = normalize_training_config({"lr_adapt_mode": "kl"})
        self.assertTrue(_post_target_optimizer_settings(14, kl_config)["override_actor_lr"])
        self.assertFalse(_post_target_optimizer_settings(15, kl_config)["override_actor_lr"])

    def test_checkpoint_separates_weights_only_from_post_target_resume(self):
        source = CTDE_PPO_Agent(3, 4, 2, 2, batch_size=512, device="cpu")
        source.training_step = 7
        source._set_critic_lr(1.5e-4)
        state = {
            "phase": "post_target",
            "post_target_episode": 100,
            "goal_index": 1,
            "consecutive_validation_passes": 1,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "post_target.pth")
            source.save(path, state)

            fresh = CTDE_PPO_Agent(
                3, 4, 2, 2, actor_lr=1e-4, critic_lr=2.5e-4, batch_size=512, device="cpu"
            )
            loaded = fresh.load(path, restore_training_state=False)
            self.assertEqual(fresh.training_step, 0)
            self.assertAlmostEqual(fresh.critic_optimizer.param_groups[0]["lr"], 2.5e-4)
            self.assertEqual(loaded["training_state"], state)

            resumed = CTDE_PPO_Agent(3, 4, 2, 2, batch_size=512, device="cpu")
            loaded = resumed.load(path, restore_training_state=True)
            self.assertEqual(resumed.training_step, 7)
            self.assertAlmostEqual(resumed.critic_optimizer.param_groups[0]["lr"], 1.5e-4)
            self.assertEqual(loaded["training_state"]["goal_index"], 1)


if __name__ == "__main__":
    unittest.main()
