import unittest

from ctde_ppo_baseline_train import _append_episode_diagnostics


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


if __name__ == "__main__":
    unittest.main()
