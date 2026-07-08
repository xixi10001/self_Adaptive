import importlib
import unittest

import numpy as np

from ctde_ppo_baseline_train import (
    CurriculumManager,
    _build_experiment_metadata,
    normalize_training_config,
)


FireEnvironmentData = importlib.import_module("\u4fe1\u606f\u8f6c\u6362").FireEnvironmentData


class AreaPercentInitializationTest(unittest.TestCase):
    def test_init_area_percent_selects_early_fire_area_and_records_stats(self):
        scene = object.__new__(FireEnvironmentData)
        scene.norm_params = {"fire_threshold": 1.0}
        scene.data = {
            "intensity": np.array(
                [
                    [2.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "time": np.array(
                [
                    [0.0, 1.0, 9.0],
                    [1.0, 1.0, 9.0],
                    [2.0, 3.0, 9.0],
                ],
                dtype=np.float32,
            ),
        }

        scene.detect_fire_boundary(time_step=0, init_area_percent=50.0)

        self.assertEqual(int(np.count_nonzero(scene.fire_binary_map)), 4)
        self.assertEqual(scene.last_boundary_sim_time, 1.0)
        self.assertEqual(
            scene.last_init_area_stats,
            {
                "total_fire_cells": 6,
                "init_fire_cells": 4,
                "actual_init_area_percent": 100.0 * 4.0 / 6.0,
                "cutoff_time": 1.0,
            },
        )


class InitAreaPercentConfigTest(unittest.TestCase):
    def test_init_area_percent_is_preferred_alias_for_init_percentile(self):
        config = normalize_training_config({"init_area_percent": 2.5})

        self.assertEqual(config["init_area_percent"], 2.5)
        self.assertEqual(config["init_percentile"], 2.5)

    def test_metadata_uav_params_config_defaults_to_baseline_behavior(self):
        default_config = normalize_training_config({})
        metadata_config = normalize_training_config({"use_metadata_uav_params": True})

        self.assertFalse(default_config["use_metadata_uav_params"])
        self.assertTrue(metadata_config["use_metadata_uav_params"])

    def test_profile_config_defaults_dims_and_validation(self):
        config = normalize_training_config({})

        self.assertEqual(config["observation_profile"], "baseline")
        self.assertEqual(config["reward_profile"], "boundary_coverage")
        self.assertEqual(
            config["observation_profile_dims"],
            {
                "baseline": {"local_obs_dim": 17, "global_state_dim": 19},
                "static_terrain": {"local_obs_dim": 24, "global_state_dim": 19},
                "dynamic_front": {"local_obs_dim": 23, "global_state_dim": 19},
                "risk_aware": {"local_obs_dim": 20, "global_state_dim": 19},
            },
        )

        configured = normalize_training_config(
            {
                "observation_profile": "dynamic_front",
                "reward_profile": "front_detection",
            }
        )
        self.assertEqual(configured["observation_profile"], "dynamic_front")
        self.assertEqual(configured["reward_profile"], "front_detection")

        with self.assertRaisesRegex(ValueError, "observation_profile"):
            normalize_training_config({"observation_profile": "unknown"})
        with self.assertRaisesRegex(ValueError, "reward_profile"):
            normalize_training_config({"reward_profile": "unknown"})

    def test_experiment_metadata_records_profiles_dataset_and_uav_params(self):
        config = normalize_training_config(
            {
                "observation_profile": "risk_aware",
                "reward_profile": "severity_weighted",
                "use_metadata_uav_params": True,
            }
        )
        dataset_index = importlib.import_module("信息转换").DatasetIndex("./dataset")
        metadata = _build_experiment_metadata(
            config,
            dataset_index,
            {
                "scene_key": "train_area001_scenario001",
                "sensor_radius_cells": 15,
                "max_steps": 800,
                "vision_radius": 15,
            },
        )

        self.assertEqual(metadata["dataset_index_version"], 2)
        self.assertEqual(
            metadata["scene_split_counts"],
            {"train": 24, "validation": 6, "generalization": 12, "stress": 4},
        )
        self.assertEqual(metadata["observation_profile"], "risk_aware")
        self.assertEqual(metadata["reward_profile"], "severity_weighted")
        self.assertEqual(metadata["norm_params_source"], "scene_p99.5")
        self.assertTrue(metadata["use_scene_uav_params"])
        self.assertEqual(metadata["sensor_radius_cells"], 15)
        self.assertEqual(metadata["max_steps"], 800)


class CurriculumScheduleTest(unittest.TestCase):
    def test_stage3_target_and_near_spawn_are_exposed(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3

        self.assertEqual(manager.current_stage3_target, 0.20)
        self.assertEqual(manager.stage3_near_prob, 0.25)

        for _ in range(125):
            manager.update(success=True, coverage=0.20)

        self.assertEqual(manager.current_stage3_target, 0.35)
        self.assertLess(manager.stage3_near_prob, 0.25)

    def test_stage3_target_does_not_advance_on_coverage_without_success(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3

        for _ in range(125):
            manager.update(success=False, coverage=0.20)

        self.assertEqual(manager.current_stage3_target, 0.20)

    def test_stage3_target_does_not_advance_with_zero_coverage_timeouts(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3

        for i in range(125):
            manager.update(success=True, coverage=0.20, zero_coverage_timeout=(i % 2 == 0))

        self.assertEqual(manager.current_stage3_target, 0.20)


if __name__ == "__main__":
    unittest.main()
