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

        self.assertEqual(config["observation_profile"], "persistent_cooperative")
        self.assertEqual(config["reward_profile"], "persistent_boundary")
        self.assertTrue(config["hierarchical_roles_enabled"])
        self.assertTrue(config["communication_enabled"])
        self.assertEqual(
            config["observation_profile_dims"],
            {
                "baseline": {"local_obs_dim": 17, "global_state_dim": 19},
                "static_terrain": {"local_obs_dim": 24, "global_state_dim": 19},
                "dynamic_front": {"local_obs_dim": 23, "global_state_dim": 19},
                "risk_aware": {"local_obs_dim": 20, "global_state_dim": 19},
                "cooperative_exploration": {"local_obs_dim": 24, "global_state_dim": 19},
                "persistent_cooperative": {"local_obs_dim": 24, "global_state_dim": 19},
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

        cooperative = normalize_training_config(
            {
                "observation_profile": "cooperative_exploration",
                "reward_profile": "novelty_search",
            }
        )
        self.assertTrue(cooperative["communication_enabled"])
        self.assertTrue(cooperative["action_mask_enabled"])
        self.assertEqual(cooperative["communication_radius_factor"], 4.0)

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
    def test_stage1_gate_uses_unique_cells_conditioned_on_boundary_found(self):
        manager = CurriculumManager()
        gate = manager.validation_gate_status(
            {
                "boundary_found_rate": 0.95,
                "zero_discovery_timeout_rate": 0.05,
                "stable_tracking_success_rate": 0.80,
                "median_first_boundary_step": 70,
                "median_unique_boundary_cells": 0.0,
                "median_unique_boundary_cells_given_found": 12.0,
            }
        )

        self.assertTrue(gate["median_unique_boundary_cells"]["passed"])
        self.assertEqual(gate["median_unique_boundary_cells"]["actual"], 12.0)

    def test_curriculum_hard_budgets_fit_3100_episode_run(self):
        total = (
            CurriculumManager.STAGE1_MAX_EPISODES
            + sum(CurriculumManager.STAGE2_MAX_EPISODES)
            + sum(CurriculumManager.STAGE3_TARGET_MAX_EPS)
            + sum(CurriculumManager.STAGE4_MAX_EPISODES)
        )
        self.assertEqual(total, 3100)
        self.assertEqual(CurriculumManager.STAGE4_MIN_REMAINING_EPISODES, 800)

    def test_stage1_budget_exhaustion_fails_instead_of_starving_later_stages(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.substage_episodes["1"] = manager.STAGE1_MAX_EPISODES
        failed = {
            "boundary_found_rate": 0.60,
            "zero_discovery_timeout_rate": 0.40,
            "stable_tracking_success_rate": 0.30,
            "median_first_boundary_step": 200,
            "median_unique_boundary_cells": 4,
        }

        manager.update_validation(failed)

        self.assertTrue(manager.curriculum_failed)
        self.assertIn("exhausted", manager.curriculum_failure_reason)

    def test_passing_validations_reach_stage4_entry_state(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        summary = {
            "boundary_found_rate": 0.95,
            "zero_discovery_timeout_rate": 0.05,
            "zero_coverage_timeout_rate": 0.05,
            "success_rate": 0.80,
            "stable_tracking_success_rate": 0.80,
            "median_first_boundary_step": 70,
            "median_unique_boundary_cells": 12,
        }

        manager.substage_episodes["1"] = manager.STAGE1_MIN_EPISODES
        for _ in range(3):
            manager.update_validation(summary)
        self.assertEqual(manager.current_substage, "2A")

        for name, minimum in zip(manager.STAGE2_SUBSTAGES, manager.STAGE2_MIN_EPISODES):
            manager.substage_episodes[name] = minimum
            for _ in range(3):
                manager.update_validation(summary)
        self.assertEqual(manager.current_substage, "3A")

        for name, minimum in zip(manager.STAGE3_SUBSTAGES, manager.STAGE3_TARGET_MIN_EPS):
            manager.substage_episodes[name] = minimum
            for _ in range(3):
                manager.update_validation(summary)

        self.assertTrue(manager._stage3_ready_for_stage4)
        self.assertTrue(manager.can_enter_stage4(summary))

    def test_unused_stage1_budget_transfers_to_stage2a(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.substage_episodes["1"] = manager.STAGE1_MIN_EPISODES
        summary = {
            "boundary_found_rate": 0.95,
            "zero_discovery_timeout_rate": 0.05,
            "success_rate": 0.80,
            "stable_tracking_success_rate": 0.80,
            "median_first_boundary_step": 70,
            "median_unique_boundary_cells_given_found": 12,
        }

        for _ in range(3):
            manager.update_validation(summary)

        expected_slack = manager.STAGE1_MAX_EPISODES - manager.STAGE1_MIN_EPISODES
        self.assertEqual(manager.current_substage, "2A")
        self.assertEqual(manager._transferable_budget, expected_slack)
        self.assertEqual(
            manager.current_budget_cap,
            manager.STAGE2_MAX_EPISODES[0] + expected_slack,
        )

    def test_stage2a_far_ratio_advances_only_on_capability_gate(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 2
        manager.substage_episodes["2A"] = manager.STAGE2_MIN_EPISODES[0]
        failed = {
            "boundary_found_rate": 0.73,
            "zero_discovery_timeout_rate": 0.27,
            "success_rate": 0.73,
            "median_first_boundary_step": 24,
        }
        passed = dict(
            failed,
            boundary_found_rate=0.85,
            zero_discovery_timeout_rate=0.10,
            success_rate=0.75,
        )

        manager.update_validation(failed)
        self.assertEqual(manager.stage2_far_spawn_ratio, 0.50)
        manager.update_validation(passed)
        self.assertEqual(manager.stage2_far_spawn_ratio, 0.80)
        manager.update_validation(passed)
        self.assertEqual(manager.stage2_far_spawn_ratio, 1.00)

    def test_stage2a_can_complete_ramp_within_base_budget(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 2
        manager.substage_episodes["2A"] = manager.STAGE2_MAX_EPISODES[0]
        passed = {
            "boundary_found_rate": 0.90,
            "zero_discovery_timeout_rate": 0.05,
            "success_rate": 0.80,
            "median_first_boundary_step": 24,
        }

        self.assertFalse(manager.update_validation(passed))
        self.assertFalse(manager.update_validation(passed))
        self.assertTrue(manager.update_validation(passed))
        self.assertEqual(manager.current_substage, "2B")

    def test_stage3_target_advances_only_after_two_of_three_validation_passes(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3

        self.assertEqual(manager.current_stage3_target, 0.20)
        self.assertEqual(manager.stage3_near_prob, 0.0)
        self.assertEqual(manager.current_substage, "3A")

        for _ in range(150):
            manager.update(success=True, coverage=0.20)

        validation_summary = {
            "boundary_found_rate": 0.90,
            "zero_coverage_timeout_rate": 0.05,
            "success_rate": 0.75,
        }
        self.assertFalse(manager.update_validation(validation_summary))
        self.assertFalse(manager.update_validation(validation_summary))
        self.assertTrue(manager.update_validation(validation_summary))
        self.assertEqual(manager.current_stage3_target, 0.35)
        self.assertEqual(manager.current_substage, "3B")

    def test_stage3_target_does_not_advance_on_failed_validation(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        manager.current_stage = 3

        for _ in range(150):
            manager.update(success=True, coverage=0.20)
        failed = {
            "boundary_found_rate": 0.90,
            "zero_coverage_timeout_rate": 0.05,
            "success_rate": 0.50,
        }
        for _ in range(3):
            self.assertFalse(manager.update_validation(failed))

        self.assertEqual(manager.current_stage3_target, 0.20)

    def test_stage1_validation_enters_stage2a_without_forced_episode_advance(self):
        manager = CurriculumManager(stage3_final_target=0.60)
        for _ in range(200):
            manager.update(success=True, coverage=0.05)
        passing = {
            "boundary_found_rate": 0.95,
            "zero_coverage_timeout_rate": 0.05,
            "stable_tracking_success_rate": 0.85,
            "median_first_boundary_step": 70,
            "median_unique_boundary_cells": 15,
        }
        for _ in range(2):
            self.assertFalse(manager.update_validation(passing))
        self.assertTrue(manager.update_validation(passing))
        self.assertEqual(manager.current_stage, 2)
        self.assertEqual(manager.current_substage, "2A")


if __name__ == "__main__":
    unittest.main()
