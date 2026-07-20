import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "dataset"
MODULE_PATH = MODULE_DIR / "\u4fe1\u606f\u8f6c\u6362.py"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

spec = importlib.util.spec_from_file_location("fire_data_module", MODULE_PATH)
fire_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fire_data_module)

env_spec = importlib.util.spec_from_file_location(
    "baseline_env_module", MODULE_DIR / "rl_environment_baseline.py"
)
baseline_env_module = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(baseline_env_module)


class FireSceneDataLoadingTest(unittest.TestCase):
    def setUp(self):
        self.dataset_index = fire_data_module.DatasetIndex(str(DATA_DIR))

    def test_scene_key_loads_complete_scene_data_from_index(self):
        scene = fire_data_module.FireSceneData(
            str(DATA_DIR),
            scene_key="train_area001_scenario001",
            dataset_index=self.dataset_index,
        )

        self.assertEqual(scene.shape, (284, 285))
        self.assertEqual(scene.static_map.shape, (8, 284, 285))
        self.assertEqual(set(scene.static_bands), set(scene.STATIC_BAND_KEYS))
        self.assertEqual(scene.sensor_radius_cells, 15)
        self.assertEqual(scene.max_steps, 800)
        self.assertEqual(scene.resolution_m, 30.0)

        for key in scene.CORE_KEYS + scene.EXTRA_RASTER_KEYS:
            self.assertIn(key, scene.data)
            self.assertEqual(scene.data[key].shape, scene.shape)
            self.assertTrue(np.isfinite(scene.data[key]).all())
            self.assertGreaterEqual(float(np.min(scene.data[key])), 0.0)

        for key in [
            "intensity_max",
            "dem_min",
            "dem_max",
            "slope_max",
            "wind_speed_max",
            "fire_threshold",
        ]:
            self.assertIn(key, scene.norm_params)

        self.assertEqual(scene.current_fire(0).shape, scene.shape)
        self.assertEqual(scene.active_front(0).shape, scene.shape)
        self.assertIsInstance(scene.boundary_points(0), list)
        self.assertEqual(scene.severity_map().shape, scene.shape)

    def test_scene_norm_params_cover_fire_rasters_and_clip_outputs(self):
        legacy_intensity_max = 600.0 + 26.94
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            scene = fire_data_module.FireSceneData(
                str(DATA_DIR),
                scene_key="train_area001_scenario001",
                dataset_index=self.dataset_index,
            )
        log_output = buffer.getvalue()

        expected_norm_keys = {
            "intensity_max",
            "length_max",
            "speedRate_max",
            "heat_per_unit_area_max",
            "crown_fire_max",
            "dem_min",
            "dem_max",
            "slope_max",
            "wind_speed_max",
            "fire_threshold",
        }
        self.assertTrue(expected_norm_keys.issubset(scene.norm_params))
        self.assertNotEqual(scene.norm_params["intensity_max"], legacy_intensity_max)
        self.assertIn("norm_params", log_output)
        self.assertIn("intensity_max", log_output)

        for key in [
            "intensity",
            "length",
            "speedRate",
            "heat_per_unit_area",
            "crown_fire",
        ]:
            normalized = scene.normalized_map(key)
            self.assertEqual(normalized.shape, scene.shape)
            self.assertGreaterEqual(float(np.min(normalized)), 0.0)
            self.assertLessEqual(float(np.max(normalized)), 1.0)

        severity = scene.severity_map()
        self.assertGreaterEqual(float(np.min(severity)), 0.0)
        self.assertLessEqual(float(np.max(severity)), 1.0)

    def test_observation_normalization_is_clipped(self):
        env = baseline_env_module.FireSearchBaselineEnvironment(
            data_dir=str(DATA_DIR),
            fixed_scene_key="train_area001_scenario001",
            max_steps=5,
            init_area_percent=5.0,
        )
        env.reset()
        scene = env.env_data
        y, x = map(int, env.drone_positions[0])
        scene.data["intensity"][y, x] = scene.norm_params["intensity_max"] * 10.0
        scene.data["dem"][y, x] = scene.norm_params["dem_max"] * 10.0
        scene.data["slope"][y, x] = scene.norm_params["slope_max"] * 10.0
        scene.data["wind_speed"][y, x] = scene.norm_params["wind_speed_max"] * 10.0
        scene.fire_binary_map[y, x] = 1

        features = env._base_cell_features(y, x)

        for key in ["intensity_norm", "dem_norm", "slope_norm", "wind_speed_norm"]:
            self.assertGreaterEqual(features[key], 0.0)
            self.assertLessEqual(features[key], 1.0)

    def test_environment_default_keeps_baseline_uav_arguments_and_profiles(self):
        env = baseline_env_module.FireSearchBaselineEnvironment(
            data_dir=str(DATA_DIR),
            fixed_scene_key="train_area001_scenario001",
            vision_radius=3,
            max_steps=5,
            init_area_percent=5.0,
        )

        obs = env.reset()
        next_obs, rewards, done, info = env.step([4, 4])

        self.assertEqual(env.vision_radius, 3)
        self.assertEqual(env.max_steps, 5)
        self.assertEqual(env.max_battery, 10)
        self.assertEqual(env.grid_size, env.env_data.shape)
        self.assertEqual(len(obs["local_obs"]), 2)
        self.assertEqual(obs["local_obs"][0].shape, (17,))
        self.assertEqual(obs["global_state"].shape, (19,))
        self.assertEqual(len(next_obs["local_obs"]), 2)
        self.assertEqual(next_obs["global_state"].shape, (19,))
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(done, bool)
        self.assertEqual(info["scene_key"], "train_area001_scenario001")

    def test_observation_profiles_have_fixed_shapes_and_step(self):
        expected_dims = {
            "baseline": 17,
            "static_terrain": 24,
            "dynamic_front": 23,
            "risk_aware": 20,
            "cooperative_exploration": 24,
        }

        for profile, local_dim in expected_dims.items():
            with self.subTest(profile=profile):
                env = baseline_env_module.FireSearchBaselineEnvironment(
                    data_dir=str(DATA_DIR),
                    fixed_scene_key="train_area001_scenario001",
                    vision_radius=3,
                    max_steps=2,
                    init_area_percent=5.0,
                    observation_profile=profile,
                )

                obs = env.reset()
                next_obs, rewards, done, info = env.step([4, 4])

                self.assertEqual(env.observation_profile, profile)
                self.assertEqual(env.local_obs_dim, local_dim)
                self.assertEqual(env.global_state_dim, 19)
                self.assertEqual(obs["local_obs"][0].shape, (local_dim,))
                self.assertEqual(obs["global_state"].shape, (19,))
                self.assertEqual(next_obs["local_obs"][0].shape, (local_dim,))
                self.assertEqual(next_obs["global_state"].shape, (19,))
                self.assertEqual(len(rewards), 2)
                self.assertIsInstance(done, bool)
                self.assertEqual(info["observation_profile"], profile)

    def test_reward_profiles_emit_standard_breakdown_keys(self):
        profiles = [
            "boundary_coverage",
            "front_detection",
            "severity_weighted",
            "exploration_balanced",
            "novelty_search",
        ]

        for profile in profiles:
            with self.subTest(profile=profile):
                env = baseline_env_module.FireSearchBaselineEnvironment(
                    data_dir=str(DATA_DIR),
                    fixed_scene_key="train_area001_scenario001",
                    vision_radius=3,
                    max_steps=1,
                    init_area_percent=5.0,
                    reward_profile=profile,
                )

                env.reset()
                _, _, done, info = env.step([4, 4])

                self.assertTrue(done)
                self.assertEqual(env.reward_profile, profile)
                self.assertEqual(info["reward_profile"], profile)
                breakdown = info["reward_breakdown"]
                for key in [
                    "r_boundary",
                    "r_front",
                    "r_severity",
                    "r_explore",
                    "r_novelty",
                    "r_revisit",
                    "r_invalid",
                    "r_overlap",
                    "r_penalty",
                ]:
                    self.assertIn(key, breakdown)
                    self.assertIsInstance(breakdown[key], float)

    def test_cooperative_exploration_shares_only_in_range_and_masks_edges(self):
        env = baseline_env_module.FireSearchBaselineEnvironment(
            data_dir=str(DATA_DIR),
            fixed_scene_key="train_area001_scenario001",
            vision_radius=3,
            max_steps=2,
            observation_profile="cooperative_exploration",
            reward_profile="novelty_search",
            communication_enabled=True,
            communication_radius_factor=4.0,
            action_mask_enabled=True,
        )
        env.reset()

        env.drone_positions = [
            np.array([50, 50], dtype=np.float32),
            np.array([50, 56], dtype=np.float32),
        ]
        env.agent_observed_masks = [np.zeros(env.grid_size, dtype=bool) for _ in range(2)]
        env.agent_known_masks = [np.zeros(env.grid_size, dtype=bool) for _ in range(2)]
        for drone_idx in range(2):
            env._mark_agent_visible_region(drone_idx, env.drone_positions[drone_idx])
        env._sync_exploration_knowledge(count_metrics=False)

        self.assertEqual(env.communication_available, [True, True])
        self.assertTrue(np.array_equal(env.agent_known_masks[0], env.agent_known_masks[1]))

        env.drone_positions = [
            np.array([0, 0], dtype=np.float32),
            np.array([50, 50], dtype=np.float32),
        ]
        env._sync_exploration_knowledge(count_metrics=False)
        self.assertEqual(env.communication_available, [False, False])

        masks = env._get_action_masks()
        self.assertEqual(masks[0].tolist(), [1, 0, 0, 1, 1])

    def test_environment_can_use_metadata_uav_params_without_per_reset_scene_log(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            env = baseline_env_module.FireSearchBaselineEnvironment(
                data_dir=str(DATA_DIR),
                fixed_scene_key="train_area001_scenario001",
                vision_radius=3,
                max_steps=5,
                init_area_percent=5.0,
                use_metadata_uav_params=True,
            )
            env.reset()

        log_output = buffer.getvalue()

        self.assertEqual(env.vision_radius, env.env_data.sensor_radius_cells)
        self.assertEqual(env.vision_radius, 15)
        self.assertEqual(env.max_steps, env.env_data.max_steps)
        self.assertEqual(env.max_steps, 800)
        self.assertEqual(env.max_battery, 1600)
        self.assertNotIn("Scene loaded |", log_output)
        self.assertNotIn("scene_key=train_area001_scenario001", log_output)

    def test_static_and_fire_shape_mismatch_reports_file_names(self):
        record = dict(self.dataset_index.get_record("train_area001_scenario001"))
        record["static_map"] = "Generalization/7/map7/map7.tif"

        with self.assertRaisesRegex(
            RuntimeError,
            r"static_map.*Generalization[\\/]7[\\/]map7[\\/]map7\.tif.*intensity.*fireline_intensity_farsite\.tif",
        ):
            fire_data_module.FireSceneData(
                str(DATA_DIR),
                scene_key="train_area001_scenario001",
                scene_record=record,
                dataset_index=self.dataset_index,
            )


if __name__ == "__main__":
    unittest.main()
