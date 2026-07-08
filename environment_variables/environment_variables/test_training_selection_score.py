import unittest

from ctde_ppo_baseline_train import _validation_model_score


class ValidationModelScoreTest(unittest.TestCase):
    def test_zero_coverage_timeout_can_make_higher_task_score_worse(self):
        high_task_unstable = {
            "mean_task_score": 0.55,
            "mean_coverage": 0.40,
            "timeout_rate": 0.70,
            "zero_coverage_timeout_rate": 0.50,
        }
        lower_task_stable = {
            "mean_task_score": 0.45,
            "mean_coverage": 0.42,
            "timeout_rate": 0.35,
            "zero_coverage_timeout_rate": 0.05,
        }

        self.assertLess(
            _validation_model_score(high_task_unstable),
            _validation_model_score(lower_task_stable),
        )


if __name__ == "__main__":
    unittest.main()
