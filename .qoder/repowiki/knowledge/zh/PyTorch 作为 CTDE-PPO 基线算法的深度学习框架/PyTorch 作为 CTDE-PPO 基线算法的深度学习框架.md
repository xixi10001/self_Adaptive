---
kind: external_dependency
name: PyTorch 作为 CTDE-PPO 基线算法的深度学习框架
slug: pytorch
category: external_dependency
category_hints:
    - framework_behavior
scope:
    - '**'
---

项目使用 PyTorch 实现 Actor-Critic 网络、PPO 更新循环与 KL 自适应学习率策略。ActorNetwork/CriticNetwork 继承自 nn.Module，使用 Categorical 分布采样动作，通过 torch.distributions 计算 log_prob/entropy，并用 nn.utils.clip_grad_norm_ 做梯度裁剪。模型以 state_dict 形式保存/加载（actor_state_dict、critic_state_dict、optimizer state）。训练时自动选择 cuda/cpu 设备，并设置 cudnn.deterministic=True 保证可复现性。