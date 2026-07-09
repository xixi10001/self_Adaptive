import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
MODULE_PATH = MODULE_DIR / "\u4fe1\u606f\u8f6c\u6362.py"
spec = importlib.util.spec_from_file_location("thermal_field_data_module", MODULE_PATH)
fire_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fire_data_module)


def _make_scene(mask: np.ndarray):
    scene = fire_data_module.FireSceneData.__new__(fire_data_module.FireSceneData)
    scene.fire_binary_map = mask.astype(np.uint8)
    intensity = np.linspace(0.0, 1.0, mask.size, dtype=np.float32).reshape(mask.shape)
    scene.data = {"intensity": intensity}
    scene.norm_params = {"intensity_max": 1.0}
    scene.thermal_field = None
    return scene


class ThermalFieldOptimizationTest(unittest.TestCase):
    def test_thermal_field_output_range_and_shape(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:28, 24:32] = 1
        scene = _make_scene(mask)

        scene._compute_thermal_field()

        self.assertEqual(scene.thermal_field.shape, mask.shape)
        self.assertGreaterEqual(float(scene.thermal_field.min()), 0.0)
        self.assertLessEqual(float(scene.thermal_field.max()), 1.0)
        self.assertIsNotNone(scene._nav_field)
        self.assertEqual(scene._nav_field.shape, mask.shape)

    def test_different_fire_masks_produce_different_fields(self):
        first_mask = np.zeros((64, 64), dtype=np.uint8)
        first_mask[8:16, 8:16] = 1
        second_mask = np.zeros((64, 64), dtype=np.uint8)
        second_mask[40:48, 40:48] = 1
        scene = _make_scene(first_mask)

        scene._compute_thermal_field()
        first_field = scene.thermal_field.copy()
        scene.fire_binary_map = second_mask
        scene._compute_thermal_field()

        self.assertFalse(np.allclose(scene.thermal_field, first_field))

    def test_no_saturation_and_gradient_exists(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[120:136, 120:136] = 1
        scene = _make_scene(mask)

        scene._compute_thermal_field()

        sat_ratio = float(np.sum(scene.thermal_field >= 0.999)) / scene.thermal_field.size
        self.assertLess(sat_ratio, 0.5, f"sat_ratio too high: {sat_ratio:.2f}")

        diag = scene.diagnose_thermal_health()
        self.assertEqual(diag["status"], "ok")
        self.assertLess(diag["zero_grad_in_high_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
