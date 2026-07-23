import importlib
import unittest

import numpy as np

from ctde_ppo_baseline_train import (
    CurriculumManager,
    _build_experiment_metadata,
    _new_validation_log,
    normalize_training_config,
)


FireEnvironmentData = importlib.import_module("\u4fe1\u606f\u8f6c\u6362").FireEnvironmentData


class ValidationLogSchemaTest(unittest.TestCase):
    def test_first_validation_can_record_stage2_curriculum_context(self):
        validation_log = _new_validation_log()

        validation_log["stage2_spawn_mix"].append([0.60, 0.30, 0.10])
        validation_log["stage2_phase"].append("early")

        self.assertEqual(
            validation_log["stage2_spawn_mix"],
            [[0.60, 0.30, 0.10]],
        )
        self.assertEqual(validation_log["stage2_phase"], ["early"])


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
    def stage1_summary(self, passing=True):
        return {
            "episodes": 30,
            "boundary_found_rate": 0.90 if passing else 0.70,
            "stage1_discovery_within_deadline_rate": 0.90 if passing else 0.70,
            "stage1_deadline_found_count": 27 if passing else 21,
            "stage1_tracking_success_count": 21 if passing else 10,
            "stage1_tracking_given_found_rate": 21 / 27 if passing else 10 / 21,
            "zero_discovery_timeout_rate": 0.10 if passing else 0.30,
            "median_unique_boundary_cells_given_found": 10.0 if passing else 5.0,
        }

    def stage2_summary(self, passing=True):
        return {
            "episodes": 30,
            "boundary_found_rate": 0.80 if passing else 0.70,
            "boundary_found_count": 24 if passing else 21,
            "stage2_found_count": 24 if passing else 21,
            "stage2_contact_success_count": 18 if passing else 12,
            "stage2_contact_given_found_rate": 0.75 if passing else 12 / 21,
            "zero_discovery_timeout_rate": 0.20 if passing else 0.30,
            "median_first_boundary_step": 200.0,
            "median_first_boundary_step_far": 230.0 if passing else 260.0,
        }

    def stage3_summary(self, rate):
        return {
            "episodes": 30,
            "boundary_found_rate": 0.85,
            "zero_discovery_timeout_rate": 0.15,
            "success_rate": rate,
            "mean_team_overlap_ratio": 0.15,
            "mean_invalid_action_count": 0.0,
        }

    def test_stage1_gate_uses_conditional_tracking_and_deadline(self):
        manager = CurriculumManager()
        gate = manager.validation_gate_status(self.stage1_summary())

        self.assertTrue(gate["boundary_found_by_150"]["passed"])
        self.assertTrue(gate["tracking_given_found"]["passed"])
        self.assertTrue(gate["median_unique_boundary_cells"]["passed"])

    def test_curriculum_soft_and_hard_paths_fit_3100_episode_run(self):
        normal = 300 + 700 + 1100 + 800
        hard = 400 + 1000 + 900 + 800

        self.assertEqual(normal, 2900)
        self.assertEqual(hard, CurriculumManager.GLOBAL_EPISODE_BUDGET)
        self.assertEqual(CurriculumManager.STAGE4_MIN_REMAINING_EPISODES, 800)

    def test_stage1_budget_exhaustion_fails_without_starving_later_stages(self):
        manager = CurriculumManager()
        manager.stage_episodes[1] = manager.STAGE1_MAX_EPISODES
        manager.substage_episodes["1"] = manager.STAGE1_MAX_EPISODES

        manager.update_validation(self.stage1_summary(False))

        self.assertTrue(manager.curriculum_failed)
        self.assertIn("hard budget", manager.curriculum_failure_reason)

    def test_three_pooled_stage1_validations_enter_single_stage2(self):
        manager = CurriculumManager()
        manager.stage_episodes[1] = manager.STAGE1_MIN_EPISODES
        manager.substage_episodes["1"] = manager.STAGE1_MIN_EPISODES

        self.assertFalse(manager.update_validation(self.stage1_summary()))
        self.assertFalse(manager.update_validation(self.stage1_summary()))
        self.assertTrue(manager.update_validation(self.stage1_summary()))
        self.assertEqual(manager.current_stage, 2)
        self.assertEqual(manager.current_substage, "2")

    def test_stage2_spawn_mix_changes_smoothly_by_episode(self):
        manager = CurriculumManager()
        manager.current_stage = 2

        self.assertEqual(manager.stage2_spawn_mix, (0.60, 0.30, 0.10))
        manager.stage_episodes[2] = 300
        self.assertEqual(manager.stage2_spawn_mix, (0.25, 0.50, 0.25))
        manager.stage_episodes[2] = 600
        self.assertEqual(manager.stage2_spawn_mix, (0.10, 0.25, 0.65))

    def test_stage2_does_not_stop_at_soft_800_but_fails_at_hard_1000(self):
        manager = CurriculumManager()
        manager.current_stage = 2
        manager.stage_episodes[2] = manager.STAGE2_SOFT_EPISODES
        manager.substage_episodes["2"] = manager.STAGE2_SOFT_EPISODES
        failed = self.stage2_summary(False)
        for _ in range(3):
            manager.update_validation(failed)
        self.assertFalse(manager.curriculum_failed)

        manager.stage_episodes[2] = manager.STAGE2_MAX_EPISODES
        manager.substage_episodes["2"] = manager.STAGE2_MAX_EPISODES
        manager.update_validation(failed)
        self.assertTrue(manager.curriculum_failed)

    def test_stage2_pass_enters_stage3_and_preserves_stage4_reserve(self):
        manager = CurriculumManager()
        manager.current_stage = 2
        manager.stage_episodes[1] = 400
        manager.stage_episodes[2] = 1000
        manager.substage_episodes["2"] = manager.STAGE2_MIN_EPISODES

        for _ in range(2):
            self.assertFalse(manager.update_validation(self.stage2_summary()))
        self.assertTrue(manager.update_validation(self.stage2_summary()))

        self.assertEqual(manager.current_stage, 3)
        self.assertEqual(sum(manager._stage3_budget_caps), 900)
        self.assertEqual(manager._stage3_budget_caps, [150, 200, 250, 300])

    def test_stage3_target_ladder_uses_decreasing_reach_rates(self):
        manager = CurriculumManager()
        manager.current_stage = 3
        manager.substage_episodes["3A"] = manager.STAGE3_TARGET_MIN_EPS[0]

        for _ in range(2):
            self.assertFalse(manager.update_validation(self.stage3_summary(0.70)))
        self.assertTrue(manager.update_validation(self.stage3_summary(0.70)))
        self.assertEqual(manager.current_stage3_target, 0.35)

    def test_passing_all_stage3_targets_reaches_stage4_entry_state(self):
        manager = CurriculumManager()
        manager.current_stage = 3
        for index, name in enumerate(manager.STAGE3_SUBSTAGES):
            manager.substage_episodes[name] = manager.STAGE3_TARGET_MIN_EPS[index]
            rate = manager.STAGE3_TARGET_REACH_RATES[index]
            for _ in range(3):
                manager.update_validation(self.stage3_summary(rate))

        self.assertTrue(manager._stage3_ready_for_stage4)
        self.assertTrue(manager.can_enter_stage4(self.stage3_summary(0.55)))


if __name__ == "__main__":
    unittest.main()
