import unittest

from ctde_ppo_baseline_train import (
    _append_episode_diagnostics,
    _assert_thermal_health_ok,
    _thermal_health_failures,
)


class TrainingDiagnosticsLogTest(unittest.TestCase):
    def test_episode_info_diagnostics_are_appended_to_training_log(self):
        training_log = {
            "avg_distance_to_fire": [],
            "first_heat_step": [],
            "first_boundary_step": [],
            "spawn_modes": [],
            "reward_breakdown": [],
        }
        info = {
            "avg_distance_to_fire": 12.5,
            "first_heat_step": 3,
            "first_boundary_step": 7,
            "spawn_modes": ["near", "far"],
            "reward_breakdown": {"r_terminal": -20.0, "r_explore": 1.5},
        }

        _append_episode_diagnostics(training_log, info)

        self.assertEqual(training_log["avg_distance_to_fire"], [12.5])
        self.assertEqual(training_log["first_heat_step"], [3])
        self.assertEqual(training_log["first_boundary_step"], [7])
        self.assertEqual(training_log["spawn_modes"], [["near", "far"]])
        self.assertEqual(
            training_log["reward_breakdown"],
            [{"r_terminal": -20.0, "r_explore": 1.5}],
        )

    def test_thermal_health_failures_report_bad_fields(self):
        records = [
            {
                "split": "train",
                "scene_key": "bad_scene",
                "sat_ratio": 0.11,
                "high_ratio": 0.10,
                "zero_grad_in_high_ratio": 0.01,
            }
        ]

        failures = _thermal_health_failures(records)

        self.assertEqual(len(failures), 1)
        self.assertIn("bad_scene", failures[0])
        self.assertIn("sat_ratio", failures[0])
        with self.assertRaises(RuntimeError):
            _assert_thermal_health_ok(records)

    def test_thermal_health_accepts_good_fields(self):
        records = [
            {
                "split": "train",
                "scene_key": "good_scene",
                "sat_ratio": 0.01,
                "high_ratio": 0.10,
                "zero_grad_in_high_ratio": 0.01,
            }
        ]

        self.assertEqual(_thermal_health_failures(records), [])
        _assert_thermal_health_ok(records)


if __name__ == "__main__":
    unittest.main()
