import os
import tempfile
import unittest

import numpy as np

from ctde_ppo_baseline_train import (
    _full_horizon_checkpoint_candidates,
    _full_horizon_episode_metrics,
    _summarize_full_horizon_records,
    normalize_training_config,
)
from rl_environment_baseline import FireSearchBaselineEnvironment


class FullHorizonEvaluationTest(unittest.TestCase):
    def make_env_state(self, termination_mode: str):
        env = FireSearchBaselineEnvironment.__new__(FireSearchBaselineEnvironment)
        env.termination_mode = termination_mode
        env.curriculum_stage = 3
        env.stage_targets = {2: 0.15, 3: 0.60}
        env.discovered_boundary = {(0, index) for index in range(6)}
        env._boundary_set = {(0, index) for index in range(10)}
        env.total_boundary_points = 10
        env.step_count = 100
        env.max_steps = 600
        env.drone_batteries = [100.0, 100.0]
        return env

    def test_target_stop_keeps_existing_mission_completion(self):
        env = self.make_env_state("target_stop")

        self.assertEqual(env._check_done(), (True, "mission_complete"))

    def test_full_horizon_ignores_target_until_horizon(self):
        env = self.make_env_state("full_horizon")

        self.assertEqual(env._check_done(), (False, "ongoing"))
        env.step_count = 600
        self.assertEqual(env._check_done(), (True, "horizon_reached"))

    def test_historical_boundary_union_coverage_uses_all_seen_boundaries(self):
        env = self.make_env_state("full_horizon")
        env.boundary_ever_mask = np.zeros((3, 4), dtype=np.bool_)
        env.confirmed_boundary_mask = np.zeros((3, 4), dtype=np.bool_)
        env.boundary_ever_mask[0, :4] = True
        env.confirmed_boundary_mask[0, :3] = True

        self.assertAlmostEqual(env._historical_boundary_union_coverage_rate(), 0.75)

    def test_full_horizon_metrics_separate_tracking_and_target_reach(self):
        metrics = _full_horizon_episode_metrics(
            [0.10, 0.20, 0.60, 0.70],
            {
                "first_target_step": 3,
                "target_reached": True,
                "historical_boundary_union_coverage": 0.80,
                "coverage_before_boundary_refresh": 0.72,
                "coverage_after_boundary_refresh": 0.70,
            },
            target=0.60,
            thresholds=[0.20, 0.60, 0.80],
            coverage_curve=[
                {"step": 0, "coverage": 0.0},
                {
                    "step": 2,
                    "coverage": 0.20,
                    "boundary_refreshed": True,
                    "coverage_before_refresh": 0.25,
                    "coverage_after_refresh": 0.20,
                },
            ],
        )

        self.assertAlmostEqual(metrics["current_coverage_auc"], 0.40)
        self.assertAlmostEqual(metrics["tail100_mean_coverage"], 0.40)
        self.assertAlmostEqual(metrics["final_current_coverage"], 0.70)
        self.assertAlmostEqual(metrics["target_hold_ratio"], 1.0)
        self.assertAlmostEqual(metrics["post_target_peak_gain"], 0.10)
        self.assertEqual(metrics["last_boundary_refresh_step"], 2)
        self.assertAlmostEqual(metrics["coverage_before_final_refresh"], 0.25)
        self.assertAlmostEqual(metrics["coverage_after_final_refresh"], 0.20)
        self.assertEqual(metrics["threshold_steps"], {"0.20": 2, "0.60": 3, "0.80": -1})

    def test_full_horizon_summary_reports_reach_and_battery_rates(self):
        records = [
            {
                "target_reached": True,
                "first_target_step": 100,
                "current_coverage_auc": 0.50,
                "tail100_mean_coverage": 0.70,
                "final_current_coverage": 0.65,
                "max_current_coverage": 0.75,
                "historical_boundary_union_coverage": 0.80,
                "target_hold_ratio": 0.60,
                "post_target_peak_gain": 0.15,
                "post_target_tail_gain": 0.10,
                "first_boundary_step": 20,
                "done_reason": "horizon_reached",
                "length": 600,
                "threshold_steps": {"0.60": 100},
            },
            {
                "target_reached": False,
                "first_target_step": -1,
                "current_coverage_auc": 0.20,
                "tail100_mean_coverage": 0.30,
                "final_current_coverage": 0.25,
                "max_current_coverage": 0.40,
                "historical_boundary_union_coverage": 0.45,
                "target_hold_ratio": 0.0,
                "post_target_peak_gain": 0.0,
                "post_target_tail_gain": 0.0,
                "first_boundary_step": -1,
                "done_reason": "battery_depleted",
                "length": 450,
                "threshold_steps": {"0.60": -1},
            },
        ]

        summary = _summarize_full_horizon_records(records, target=0.60)

        self.assertAlmostEqual(summary["target_reach_rate"], 0.50)
        self.assertAlmostEqual(summary["mean_current_coverage_auc"], 0.35)
        self.assertAlmostEqual(summary["horizon_completion_rate"], 0.50)
        self.assertAlmostEqual(summary["battery_depleted_rate"], 0.50)
        self.assertEqual(summary["mean_first_target_step"], 100.0)

    def test_terminal_focus_checkpoint_candidates_only_use_late_stage3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in [
                "ppo_ep2100_stage3.pth",
                "ppo_ep2200_stage3.pth",
                "ppo_ep2300_stage3.pth",
                "ppo_ep2400_stage2.pth",
            ]:
                with open(os.path.join(temp_dir, filename), "wb") as handle:
                    handle.write(b"checkpoint")

            candidates = _full_horizon_checkpoint_candidates(temp_dir, final_episode=2500)

        self.assertEqual([name for name, _ in candidates], ["ep2200", "ep2300"])

    def test_default_config_enables_separate_full_horizon_evaluation(self):
        config = normalize_training_config()

        self.assertEqual(config["evaluation_mode"], "target_stop")
        self.assertTrue(config["full_horizon_eval_after_train"])
        self.assertEqual(config["full_horizon_curve_stride"], 20)
        self.assertEqual(config["full_horizon_thresholds"], [0.20, 0.40, 0.60, 0.80])


if __name__ == "__main__":
    unittest.main()
