import unittest
from types import SimpleNamespace

import numpy as np

from ctde_ppo_baseline_train import CurriculumManager
from rl_environment_baseline import (
    CooperativeTaskCoordinator,
    FireSearchBaselineEnvironment,
)


def _failing_stage1_summary(score_level: float = 0.5):
    score_level = float(score_level)
    return {
        "episodes": 30,
        "boundary_found_rate": score_level,
        "stage1_discovery_within_deadline_rate": score_level,
        "stage1_deadline_found_count": int(round(30 * score_level)),
        "stage1_tracking_given_found_rate": min(score_level, 0.69),
        "stage1_tracking_success_count": int(
            round(30 * score_level * min(score_level, 0.69))
        ),
        "zero_discovery_timeout_rate": 1.0 - score_level,
        "median_unique_boundary_cells": 8.0,
        "median_unique_boundary_cells_given_found": 8.0,
    }


class SearchTargetProgressRewardTest(unittest.TestCase):
    def make_env(self):
        env = FireSearchBaselineEnvironment.__new__(
            FireSearchBaselineEnvironment
        )
        coordinator = SimpleNamespace(
            SEARCH=CooperativeTaskCoordinator.SEARCH,
            current_roles=[CooperativeTaskCoordinator.SEARCH],
            assigned_task_valid=[True],
            assigned_targets=[np.array([0.0, 5.0], dtype=np.float32)],
            search_target_steps=[0],
        )
        env.task_coordinator = coordinator
        env.discovered_boundary = set()
        env.step_count = 0
        env.search_target_refresh_steps = 20
        env.vision_radius = 2
        env.search_target_progress_reward_weight = 0.05
        env.search_target_progress_step_cap = 0.05
        env.search_target_progress_episode_cap = 2.0
        env._episode_search_target_progress_total = 0.0
        env.search_target_progress_eligible_steps = 0
        env.search_target_follow_steps = 0
        return env

    def test_progress_reward_is_positive_bounded_and_deployable(self):
        env = self.make_env()
        reward = env._search_target_progress_reward(
            0,
            np.array([0, 0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )
        self.assertAlmostEqual(reward, 0.05)
        self.assertEqual(env.search_target_progress_eligible_steps, 1)
        self.assertEqual(env.search_target_follow_steps, 1)

        total = reward
        for _ in range(100):
            total += env._search_target_progress_reward(
                0,
                np.array([0, 0], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
            )
        self.assertLessEqual(total, 2.0 + 1e-9)

        env.discovered_boundary.add((1, 1))
        self.assertEqual(
            env._search_target_progress_reward(
                0,
                np.array([0, 0], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
            ),
            0.0,
        )


class PreBoundaryPenaltyFloorTest(unittest.TestCase):
    def test_revisit_and_overlap_penalties_have_episode_floors(self):
        env = FireSearchBaselineEnvironment.__new__(
            FireSearchBaselineEnvironment
        )
        env.grid_size = (8, 8)
        env.num_drones = 2
        env.vision_radius = 2
        env.agent_observed_masks = [
            np.ones(env.grid_size, dtype=np.bool_) for _ in range(2)
        ]
        env.pre_boundary_team_novelty_weight = 0.20
        env.pre_boundary_unique_novelty_weight = 0.10
        env.pre_boundary_overlap_weight = 0.10
        env.pre_boundary_revisit_weight = 0.03
        env.pre_boundary_reward_episode_cap = 6.0
        env.pre_boundary_revisit_penalty_floor = -8.0
        env.pre_boundary_overlap_penalty_floor = -4.0
        env._episode_pre_boundary_reward_total = 0.0
        env._episode_pre_boundary_revisit_penalty_total = 0.0
        env._episode_pre_boundary_overlap_penalty_total = 0.0
        env.pre_boundary_agent_steps = 0
        env.pre_boundary_revisit_steps = 0.0
        env.team_overlap_sum = 0.0

        positions = [
            np.array([3, 3], dtype=np.int32),
            np.array([3, 3], dtype=np.int32),
        ]
        for _ in range(500):
            env._pre_boundary_cooperative_search_reward(positions)

        self.assertGreaterEqual(
            env._episode_pre_boundary_revisit_penalty_total, -8.0 - 1e-9
        )
        self.assertGreaterEqual(
            env._episode_pre_boundary_overlap_penalty_total, -4.0 - 1e-9
        )


class Stage1SoftBudgetTest(unittest.TestCase):
    def make_manager(self, episodes=400):
        manager = CurriculumManager()
        manager.stage_episodes[1] = episodes
        manager.substage_episodes["1"] = episodes
        return manager

    def test_resource_floor_blocks_early_plateau_failure(self):
        manager = self.make_manager()
        summary = _failing_stage1_summary(0.5)
        for _ in range(6):
            manager.update_validation(
                summary,
                total_environment_steps=64000,
                ppo_updates=14,
            )
        self.assertFalse(manager.curriculum_failed)

        manager.update_validation(
            summary,
            total_environment_steps=65000,
            ppo_updates=15,
        )
        self.assertTrue(manager.curriculum_failed)
        self.assertIn("plateaued", manager.curriculum_failure_reason)

    def test_three_clear_degradations_request_best_checkpoint_rollback(self):
        manager = self.make_manager(episodes=300)
        good = _failing_stage1_summary(0.80)
        bad = _failing_stage1_summary(0.40)
        for _ in range(3):
            manager.update_validation(
                good,
                total_environment_steps=50000,
                ppo_updates=12,
            )
        for _ in range(3):
            manager.update_validation(
                bad,
                total_environment_steps=50000,
                ppo_updates=12,
            )
        self.assertTrue(manager._stage1_rollback_requested)
        self.assertFalse(manager.curriculum_failed)

    def test_stage1_hard_budget_is_six_hundred_episodes(self):
        manager = self.make_manager(episodes=600)
        summary = _failing_stage1_summary(0.5)
        for _ in range(3):
            manager.update_validation(
                summary,
                total_environment_steps=90000,
                ppo_updates=20,
            )
        self.assertTrue(manager.curriculum_failed)
        self.assertIn("600-episode", manager.curriculum_failure_reason)


if __name__ == "__main__":
    unittest.main()
