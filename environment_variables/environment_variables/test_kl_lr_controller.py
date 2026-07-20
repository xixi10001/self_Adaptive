import os
import tempfile
import unittest

import numpy as np
import torch

from ctde_ppo_baseline_train import CTDE_PPO_Agent, normalize_training_config


class KLLearningRateControllerTest(unittest.TestCase):
    def make_agent(self, **overrides):
        kwargs = {
            "local_obs_dim": 3,
            "global_state_dim": 4,
            "action_dim": 2,
            "num_agents": 2,
            "actor_lr": 2e-4,
            "critic_lr": 5e-4,
            "lr_adapt_mode": "kl",
            "target_kl": 0.01,
            "actor_lr_min": 1e-4,
            "actor_lr_max": 2.5e-4,
            "kl_ema_beta": 0.0,
            "batch_size": 512,
            "device": "cpu",
        }
        kwargs.update(overrides)
        return CTDE_PPO_Agent(**kwargs)

    def test_default_config_uses_bounded_hysteresis_controller(self):
        config = normalize_training_config()

        self.assertEqual(config["target_kl"], 0.0065)
        self.assertEqual(config["actor_lr_min"], 1e-4)
        self.assertEqual(config["actor_lr_max"], 2.5e-4)
        self.assertEqual(config["kl_ema_beta"], 0.8)
        self.assertEqual(config["kl_lr_low_ratio"], 0.82)
        self.assertEqual(config["kl_lr_down_factor"], 0.90)
        self.assertEqual(config["kl_lr_low_patience"], 3)
        self.assertEqual(config["kl_early_stop_ratio"], 1.5)

    def test_low_kl_requires_patience_before_lr_increase(self):
        agent = self.make_agent()

        self.assertEqual(agent._adapt_actor_lr_by_kl(0.007), "low_wait")
        self.assertEqual(agent._adapt_actor_lr_by_kl(0.007), "low_wait")
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 2e-4)

        self.assertEqual(agent._adapt_actor_lr_by_kl(0.007), "up")
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 2.06e-4)
        self.assertEqual(agent._consecutive_low_kl, 0)

    def test_emergency_actor_reduction_does_not_change_critic_lr(self):
        agent = self.make_agent()
        agent._consecutive_low_kl = 2

        action = agent._adapt_actor_lr_by_kl(0.021)

        self.assertEqual(action, "emergency_down")
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 1.4e-4)
        self.assertAlmostEqual(agent.critic_optimizer.param_groups[0]["lr"], 5e-4)
        self.assertEqual(agent._consecutive_low_kl, 0)

    def test_regular_high_kl_reduction_is_not_over_aggressive(self):
        agent = self.make_agent()

        action = agent._adapt_actor_lr_by_kl(0.013)

        self.assertEqual(action, "down")
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 1.8e-4)

    def test_checkpoint_restores_controller_only_for_training_resume(self):
        source = self.make_agent(target_kl=0.0065)
        source.kl_ema = 0.004
        source._consecutive_low_kl = 2
        source._set_actor_lr(1.7e-4)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "agent.pth")
            source.save(checkpoint_path)

            resumed = self.make_agent(target_kl=0.02)
            resumed.load(checkpoint_path, restore_training_state=True)
            self.assertEqual(resumed.target_kl, 0.0065)
            self.assertEqual(resumed.kl_ema, 0.004)
            self.assertEqual(resumed._consecutive_low_kl, 2)
            self.assertAlmostEqual(resumed.actor_optimizer.param_groups[0]["lr"], 1.7e-4)

            evaluation = self.make_agent(actor_lr=2.2e-4, target_kl=0.02)
            evaluation.load(checkpoint_path, restore_training_state=False)
            self.assertEqual(evaluation.target_kl, 0.02)
            self.assertIsNone(evaluation.kl_ema)
            self.assertEqual(evaluation._consecutive_low_kl, 0)
            self.assertAlmostEqual(evaluation.actor_optimizer.param_groups[0]["lr"], 2.2e-4)

    def test_resume_clips_actor_lr_from_legacy_checkpoint(self):
        source = self.make_agent(actor_lr=4e-4, actor_lr_max=4e-4)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "legacy_lr.pth")
            source.save(checkpoint_path)

            resumed = self.make_agent(actor_lr_max=2.5e-4)
            resumed.load(checkpoint_path, restore_training_state=True)

        self.assertAlmostEqual(resumed.actor_optimizer.param_groups[0]["lr"], 2.5e-4)

    def test_kl_mode_stops_remaining_ppo_epochs_without_decaying_critic_lr(self):
        agent = self.make_agent(target_kl=1e-8, ppo_epochs=4)
        rng = np.random.default_rng(7)
        local_obs = rng.normal(size=(512, 2, 3)).astype(np.float32)
        flat_obs = torch.from_numpy(local_obs.reshape(-1, 3))
        with torch.no_grad():
            distribution = agent.actor.get_action_probs(flat_obs)
            actions = distribution.sample().reshape(512, 2)
            log_probs = distribution.log_prob(actions.reshape(-1)).reshape(512, 2)

        global_states = rng.normal(size=(512, 4)).astype(np.float32)
        rewards = rng.normal(size=(512, 2)).astype(np.float32)
        for index in range(512):
            agent.store_transition(
                local_obs[index],
                global_states[index],
                actions[index].tolist(),
                log_probs[index].tolist(),
                rewards[index].tolist(),
                (index + 1) % 64 == 0,
            )

        update_info = agent.update(force=True)

        self.assertTrue(update_info["kl_early_stop"])
        self.assertLess(update_info["ppo_epochs_completed"], 4)
        self.assertAlmostEqual(update_info["critic_lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
