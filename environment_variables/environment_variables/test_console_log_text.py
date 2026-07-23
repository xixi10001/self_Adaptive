import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ctde_ppo_baseline_train import CurriculumManager, _run_figure_scripts


class ConsoleLogTextTest(unittest.TestCase):
    def test_curriculum_stage_switch_log_uses_chinese_labels(self):
        manager = CurriculumManager()
        manager.substage_episodes["1"] = 200
        passing = {
            "episodes": 20,
            "boundary_found_rate": 0.95,
            "stage1_discovery_within_deadline_rate": 0.95,
            "stage1_deadline_found_count": 19,
            "stage1_tracking_success_count": 17,
            "zero_discovery_timeout_rate": 0.05,
            "stable_tracking_success_rate": 0.85,
            "median_first_boundary_step": 70,
            "median_unique_boundary_cells": 15,
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for _ in range(3):
                manager.update_validation(passing)

        text = output.getvalue()
        self.assertIn("验证课程", text)
        self.assertIn("1 -> 2", text)
        self.assertIn("pooled_validations=3/3", text)
        self.assertNotIn("Curriculum stage", text)

    def test_figure_runner_skips_missing_generalization_data(self):
        config = {"figure_window": 10, "figure_dpi": 100, "max_steps": 600}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "ctde_ppo_baseline_train.subprocess.run"
        ) as run_mock, contextlib.redirect_stdout(output):
            paths = _run_figure_scripts(
                tmpdir,
                config,
                include_generalization=True,
                out_root=str(Path(tmpdir) / "figures"),
            )

        self.assertEqual(run_mock.call_count, 1)
        self.assertIn("training_figures", paths)
        self.assertNotIn("generalization_figures", paths)
        self.assertIn("跳过泛化评估图表", output.getvalue())


if __name__ == "__main__":
    unittest.main()
