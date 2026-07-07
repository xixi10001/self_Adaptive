import os
import tempfile
import unittest

from ctde_ppo_baseline_train import (
    _after_train_eval_checkpoints,
    _eval_summary_stage,
    normalize_training_config,
)


class BestValOnlyEvalTest(unittest.TestCase):
    def test_after_train_eval_checkpoints_only_returns_existing_best_val(self):
        config = normalize_training_config({})

        with tempfile.TemporaryDirectory() as tmpdir:
            best_val_path = os.path.join(tmpdir, "ppo_best_val.pth")
            with open(best_val_path, "wb") as f:
                f.write(b"checkpoint")

            checkpoints = _after_train_eval_checkpoints(
                config,
                {
                    "final": os.path.join(tmpdir, "ppo_final.pth"),
                    "best_val": best_val_path,
                },
            )

        self.assertEqual(checkpoints, [("best_val", best_val_path)])

    def test_eval_summary_stage_prefers_best_val_over_final(self):
        eval_summary = {
            "final": {
                "generalization": {
                    "stages": {
                        "3": {"mean_task_score": 0.1},
                    },
                },
            },
            "best_val": {
                "splits": {
                    "generalization": {
                        "stages": {
                            "3": {"mean_task_score": 0.3},
                        },
                    },
                },
            },
        }

        summary = _eval_summary_stage(eval_summary, "generalization", "3")

        self.assertEqual(summary["mean_task_score"], 0.3)


if __name__ == "__main__":
    unittest.main()
