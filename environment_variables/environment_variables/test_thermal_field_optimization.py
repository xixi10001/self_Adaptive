import importlib.util
from pathlib import Path
import unittest
from unittest import mock

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
    scene.thermal_field = None
    return scene


class ThermalFieldOptimizationTest(unittest.TestCase):
    def test_identical_fire_mask_reuses_cached_blur(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:28, 24:32] = 1
        scene = _make_scene(mask)

        with mock.patch.object(
            fire_data_module,
            "gaussian_filter",
            wraps=fire_data_module.gaussian_filter,
        ) as mocked_filter:
            scene._compute_thermal_field()
            first_field = scene.thermal_field.copy()
            scene._compute_thermal_field()

        self.assertEqual(mocked_filter.call_count, 1)
        np.testing.assert_array_equal(scene.thermal_field, first_field)

    def test_equal_count_masks_at_different_positions_do_not_share_cache_entry(self):
        first_mask = np.zeros((64, 64), dtype=np.uint8)
        first_mask[8:16, 8:16] = 1
        second_mask = np.zeros((64, 64), dtype=np.uint8)
        second_mask[40:48, 40:48] = 1
        scene = _make_scene(first_mask)

        with mock.patch.object(
            fire_data_module,
            "gaussian_filter",
            wraps=fire_data_module.gaussian_filter,
        ) as mocked_filter:
            scene._compute_thermal_field()
            first_field = scene.thermal_field.copy()
            scene.fire_binary_map = second_mask
            scene._compute_thermal_field()

        self.assertEqual(mocked_filter.call_count, 2)
        self.assertEqual(len(scene._thermal_field_cache), 2)
        self.assertFalse(np.allclose(scene.thermal_field, first_field))

    def test_cache_stores_quarter_resolution_blur_and_returns_full_resolution_field(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        scene = _make_scene(mask)

        scene._compute_thermal_field()

        cached_field = next(iter(scene._thermal_field_cache.values()))
        self.assertEqual(cached_field.shape, (16, 16))
        self.assertEqual(scene.thermal_field.shape, mask.shape)
        self.assertGreaterEqual(float(scene.thermal_field.min()), 0.0)
        self.assertLessEqual(float(scene.thermal_field.max()), 100.0)


if __name__ == "__main__":
    unittest.main()
