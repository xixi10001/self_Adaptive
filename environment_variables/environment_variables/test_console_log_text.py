import contextlib
import io
import unittest

from ctde_ppo_baseline_train import CurriculumManager


class ConsoleLogTextTest(unittest.TestCase):
    def test_curriculum_stage_switch_log_uses_chinese_labels(self):
        manager = CurriculumManager()
        manager.stage_min_episodes[1] = 1
        manager.stage_thresholds[1] = 0.0

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manager.update(success=True, coverage=0.25)

        text = output.getvalue()
        self.assertIn("课程阶段 1 -> 2", text)
        self.assertIn("本阶段回合=1", text)
        self.assertIn("成功率=100.0%", text)
        self.assertIn("覆盖率=25.0%", text)
        self.assertNotIn("Curriculum stage", text)


if __name__ == "__main__":
    unittest.main()
