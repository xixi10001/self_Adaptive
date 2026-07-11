# PPO算法核心实现

<cite>
**本文引用的文件**   
- [ctde_ppo_baseline_train.py](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py)
</cite>

## 更新摘要
**所做更改**   
- 增强了CTDE_PPO_Agent类的内部状态跟踪机制，新增_consecutive_low_kl计数器用于耐心机制
- 改进了检查点保存功能，现在能够完整保存学习率控制器状态
- 增强了日志记录功能，新增了target_kl、consecutive_low_kl、kl_early_stop和ppo_epochs_completed等指标
- 更新了KL自适应学习率算法，实现了更精细的学习率调整策略

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能与数值稳定性](#性能与数值稳定性)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录：超参数影响与调参建议](#附录超参数影响与调参建议)

## 简介
本技术文档聚焦于仓库中PPO（近端策略优化）算法的核心实现，系统阐述以下要点：
- 优势函数计算与GAE（Generalized Advantage Estimation）的推导、实现细节与数值稳定性处理
- PPO目标函数的构造与裁剪机制防止策略更新过大的数学原理
- 经验回放缓冲区ReplayBuffer的数据结构设计、采样策略与批处理优化
- 完整训练循环：前向传播、损失计算、反向传播与参数更新
- KL散度监控、梯度裁剪与数值稳定性的工程实践
- 关键超参数clip_epsilon、gae_lambda、entropy_coef对训练稳定性和收敛性的影响
- **新增**：增强的KL自适应学习率控制机制，包括耐心机制和早停策略

## 项目结构
该仓库包含大量场景数据与输出结果，PPO算法核心逻辑集中在单一训练脚本中。主要模块包括：
- 网络模型：ActorNetwork（策略）、CriticNetwork（价值）
- 经验缓冲：ReplayBuffer
- 智能体与控制流：CTDE_PPO_Agent（封装GAE、PPO更新、KL自适应学习率等）
- 训练主循环与评估：train、evaluate、run_lr_comparison、main

```mermaid
graph TB
A["训练入口 main()"] --> B["训练流程 train()"]
B --> C["环境交互: 收集轨迹到 ReplayBuffer"]
C --> D["GAE 计算 compute_gae()"]
D --> E["PPO 更新 update()"]
E --> F["Actor 网络 ActorNetwork"]
E --> G["Critic 网络 CriticNetwork"]
E --> H["KL 监控与自适应学习率"]
H --> I["耐心机制 _consecutive_low_kl"]
H --> J["早停策略 kl_early_stop"]
B --> K["验证与保存模型"]
B --> L["训练后评估 evaluate()"]
```

图表来源
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)
- [ctde_ppo_baseline_train.py:779-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L779-L1097)

章节来源
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)
- [ctde_ppo_baseline_train.py:779-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L779-L1097)

## 核心组件
- ActorNetwork：多层全连接+LayerNorm+残差块，输出离散动作logits，提供概率分布接口
- CriticNetwork：多层全连接+LayerNorm+残差块，输出标量状态价值
- ReplayBuffer：按时间步顺序存储本地观测、全局状态、动作、旧对数概率、奖励、终止标记
- CTDE_PPO_Agent：封装选择动作、GAE计算、PPO多轮小批量更新、KL自适应学习率、梯度裁剪、指标统计

**更新** CTDE_PPO_Agent类现在包含增强的内部状态跟踪机制，支持更精细的学习率控制和训练监控。

章节来源
- [ctde_ppo_baseline_train.py:460-535](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L460-L535)
- [ctde_ppo_baseline_train.py:537-566](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L537-L566)
- [ctde_ppo_baseline_train.py:779-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L779-L1097)

## 架构总览
下图展示PPO在训练中的关键调用序列与数据流向：

```mermaid
sequenceDiagram
participant T as "训练循环"
participant A as "Agent.select_actions()"
participant E as "环境 step()"
participant R as "ReplayBuffer.store()"
participant U as "Agent.update()"
participant G as "compute_gae()"
participant AC as "ActorNetwork"
participant CR as "CriticNetwork"
participant KL as "KL自适应控制器"
T->>A : 输入 local_obs
A-->>T : actions, old_log_probs
T->>E : 执行 actions
E-->>T : next_obs, rewards, done, info
T->>R : store(local_obs, global_state, actions, log_probs, rewards, done)
alt 缓冲区达到批次大小
T->>U : 触发更新
U->>G : 计算 advantages, returns
G->>CR : 前向得到 values
U->>AC : 新策略分布
U->>U : 裁剪目标 + 熵正则
U->>AC : 反向传播 + 梯度裁剪 + 更新
U->>CR : 反向传播 + 梯度裁剪 + 更新
U->>KL : 计算KL并调整学习率
KL->>KL : 更新_consecutive_low_kl计数
KL->>KL : 检查早停条件
U-->>T : 返回指标(含approx_kl, clip_fraction, ppo_epochs_completed)
end
```

图表来源
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)
- [ctde_ppo_baseline_train.py:779-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L779-L1097)

## 详细组件分析

### 优势函数与GAE实现
- 团队奖励聚合：将多智能体每步奖励取均值作为团队奖励，降低方差并统一信号
- TD误差delta：使用当前值估计V(s_t)、下一时刻值V(s_{t+1})与折扣因子gamma计算
- GAE递归：从最后一步逆序累积，结合终止标志dones控制回传边界
- 数值稳定：advantages标准化为均值为0、标准差为1（加极小常数防除零）

```mermaid
flowchart TD
Start(["进入 compute_gae"]) --> V["用 Critic 前向得到 values"]
V --> TeamRew["团队奖励 = mean(rewards)"]
TeamRew --> Init["gae=0; advantages=[]; returns=[]"]
Init --> Loop{"逆序遍历 t"}
Loop --> |t=最后一步| NextVal0["next_value=0"]
Loop --> |其他| NextVal["next_value=values[t+1]"]
NextVal0 --> Delta["delta = r_t + gamma*next_value*(1-d_t) - V_t"]
NextVal --> Delta
Delta --> GAE["gae = delta + gamma*lambda*(1-d_t)*gae"]
GAE --> InsertAdv["advantages.insert(0, gae)"]
InsertAdv --> InsertRet["returns.insert(0, gae + V_t)"]
InsertRet --> Loop
Loop --> End(["返回 advantages, returns"])
```

图表来源
- [ctde_ppo_baseline_train.py:921-941](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L921-L941)

章节来源
- [ctde_ppo_baseline_train.py:921-941](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L921-L941)

### PPO目标函数与裁剪机制
- 重要性采样比率ratio = exp(log π_θ(a|s) - log π_θ_old(a|s))
- 无裁剪目标 surr1 = ratio * A
- 裁剪目标 surr2 = clip(ratio, 1-ε, 1+ε) * A
- 最终actor目标：min(surr1, surr2) 的负均值，减去熵正则项 entropy_coef * H(π_θ)
- 裁剪的数学动机：限制新旧策略分布偏离程度，避免单次更新过大导致性能崩溃；通过"保守"目标保证单调改进近似

```mermaid
flowchart TD
S(["开始"]) --> LogRatio["log_ratio = new_log_prob - old_log_prob"]
LogRatio --> Ratio["ratio = exp(log_ratio)"]
Ratio --> Surr1["surr1 = ratio * A"]
Ratio --> Clip["clip(ratio, 1-ε, 1+ε)"]
Clip --> Surr2["surr2 = clipped_ratio * A"]
Surr1 --> Min["loss = -min(surr1, surr2)"]
Surr2 --> Min
Min --> Entropy["entropy = H(π_θ)"]
Entropy --> Final["actor_loss = loss - α * entropy"]
Final --> E(["结束"])
```

图表来源
- [ctde_ppo_baseline_train.py:999-1007](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L999-L1007)

章节来源
- [ctde_ppo_baseline_train.py:999-1007](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L999-L1007)

### 经验回放缓冲区 ReplayBuffer
- 数据结构：以列表形式按时间步顺序存放local_obs、global_states、actions、log_probs、rewards、dones
- 写入接口：store(...) 追加单步数据
- 读取接口：get() 一次性返回全部历史
- 清理接口：clear() 清空所有列表
- 长度接口：__len__() 基于rewards长度
- 设计权衡：简单直观，适合离线GAE与整段轨迹更新；内存占用随episode长度线性增长

章节来源
- [ctde_ppo_baseline_train.py:537-566](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L537-L566)

### 采样策略与批处理优化
- 整段轨迹先用于GAE计算，再打乱索引进行多轮epoch的小批量训练
- mini_batch_size默认取batch_size//8且不低于512，提高GPU吞吐与稳定性
- 每次迭代内分别更新critic与actor，critic使用MSE回归到returns，actor使用PPO裁剪目标
- 多智能体维度展平：将(num_agents,)维度的动作与优势展平到样本维度，确保与网络输出对齐

章节来源
- [ctde_ppo_baseline_train.py:968-1023](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L968-L1023)

### 训练循环与参数更新
- 交互阶段：每步选择动作、记录旧对数概率与环境反馈，存入ReplayBuffer
- 更新触发：当缓冲区长度≥batch_size时触发update()；支持force=True强制更新
- 更新过程：
  - 计算advantages与returns并进行标准化
  - ppo_epochs轮内随机打乱索引，按mini_batch_size切片
  - critic：MSE损失，梯度裁剪max_grad_norm，Adam更新
  - actor：PPO裁剪目标+熵正则，梯度裁剪，Adam更新
  - 统计approx_kl、clip_fraction、entropy等指标
  - **新增**：跟踪ppo_epochs_completed和kl_early_stop状态
- 结束后清空缓冲区，累计training_step

**更新** 训练循环现在支持早停机制，当KL散度超过阈值时可以提前停止当前epoch的训练。

```mermaid
sequenceDiagram
participant L as "训练循环"
participant B as "ReplayBuffer"
participant U as "Agent.update()"
participant C as "Critic"
participant A as "Actor"
participant KL as "KL控制器"
L->>B : store(每步数据)
alt 缓冲区足够
L->>U : update(force=False)
U->>U : GAE + 标准化
loop ppo_epochs
U->>C : 前向预测V，MSE损失，裁剪梯度，更新
U->>A : 计算ratio/surr，裁剪目标，熵正则，裁剪梯度，更新
U->>U : 统计KL、clip_fraction等
U->>KL : 检查早停条件
alt KL超过阈值
KL-->>U : kl_early_stop = True
U-->>L : 提前退出当前epoch
else 继续训练
KL-->>U : 继续下一个epoch
end
end
U->>B : clear()
U-->>L : 返回指标(含ppo_epochs_completed, kl_early_stop)
end
```

图表来源
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)
- [ctde_ppo_baseline_train.py:943-1060](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L943-L1060)

章节来源
- [ctde_ppo_baseline_train.py:943-1060](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L943-L1060)
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)

### KL散度监控与自适应学习率

**重大更新** CTDE_PPO_Agent类实现了全新的KL自适应学习率控制机制，包含以下增强功能：

#### 耐心机制（Patience Mechanism）
- `_consecutive_low_kl`计数器：跟踪连续低KL状态的持续时间
- `kl_lr_low_patience`参数：设置触发学习率增加的耐心阈值（默认3次）
- 当KL持续低于目标值的80%时，计数器递增，达到耐心阈值后才增加学习率

#### 多级学习率调整策略
- **紧急下降**：当KL > 2.0 × target_kl时，学习率乘以0.70倍
- **正常下降**：当KL > 1.2 × target_kl时，学习率乘以0.85倍  
- **耐心等待**：当KL < 0.8 × target_kl时，等待耐心阈值
- **学习率提升**：满足耐心条件后，学习率乘以1.03倍
- **保持状态**：KL在合理范围内时保持不变

#### 早停机制（Early Stopping）
- `kl_early_stop_ratio`参数：设置早停阈值（默认1.50 × target_kl）
- 在每个epoch结束时检查运行平均KL是否超过阈值
- 如果超过阈值，设置`kl_early_stop = True`并提前退出当前epoch

#### 增强的日志记录
- `target_kl`：当前的KL目标值
- `consecutive_low_kl`：连续低KL状态的计数
- `kl_early_stop`：是否触发了早停
- `ppo_epochs_completed`：实际完成的epoch数量
- `kl_lr_action`：学习率调整动作（up/down/emergency_down/keep/low_wait/fixed）

```mermaid
flowchart TD
Start(["KL自适应学习率"]) --> UpdateEMA["_update_kl_ema(mean_kl)"]
UpdateEMA --> CheckEmergency{mean_kl > 2.0 × target_kl?}
CheckEmergency --> |是| EmergencyDown[_consecutive_low_kl = 0<br/>lr *= 0.70<br/>action = emergency_down]
CheckEmergency --> |否| CheckHigh{max(mean_kl, kl_ema) > 1.2 × target_kl?}
CheckHigh --> |是| NormalDown[_consecutive_low_kl = 0<br/>lr *= 0.85<br/>action = down]
CheckHigh --> |否| CheckLow{kl_ema < 0.8 × target_kl?}
CheckLow --> |是| IncrementCounter[_consecutive_low_kl += 1]
IncrementCounter --> CheckPatience{_consecutive_low_kl >= patience?}
CheckPatience --> |是| IncreaseLR[_consecutive_low_kl = 0<br/>lr *= 1.03<br/>action = up]
CheckPatience --> |否| Wait[保持当前lr<br/>action = low_wait]
CheckLow --> |否| Keep[保持当前lr<br/>action = keep]
EmergencyDown --> SetLR[_set_actor_lr(new_lr)]
NormalDown --> SetLR
IncreaseLR --> SetLR
Wait --> SetLR
Keep --> SetLR
SetLR --> ReturnAction[返回action]
```

图表来源
- [ctde_ppo_baseline_train.py:873-901](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L873-L901)

章节来源
- [ctde_ppo_baseline_train.py:873-901](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L873-L901)
- [ctde_ppo_baseline_train.py:1025-1032](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1025-L1032)

### 检查点保存与恢复

**重大更新** 检查点机制现在能够完整保存学习率控制器的内部状态：

#### 增强的保存功能
- 保存完整的`lr_controller_state`字典，包含：
  - `kl_ema`：KL指数移动平均值
  - `consecutive_low_kl`：连续低KL计数
  - `target_kl`：当前目标KL值
- 向后兼容：仍然支持旧的保存格式

#### 智能恢复机制
- 优先从`lr_controller_state`加载控制器状态
- 如果不存在则回退到旧的`kl_ema`字段
- 支持`restore_training_state`参数控制是否恢复训练状态
- 自动同步当前学习率与优化器状态

```mermaid
flowchart TD
Save(["保存检查点"]) --> SaveState["保存完整状态"]
SaveState --> ActorState["actor_state_dict"]
SaveState --> CriticState["critic_state_dict"]
SaveState --> OptimizerStates["optimizer_state_dicts"]
SaveState --> TrainingStep["training_step"]
SaveState --> LRController["lr_controller_state:<br/>- kl_ema<br/>- consecutive_low_kl<br/>- target_kl"]
Load(["加载检查点"]) --> LoadCheckpoint["加载checkpoint"]
LoadCheckpoint --> RestoreMode{restore_training_state?}
RestoreMode --> |否| ResetState[kl_ema = None<br/>_consecutive_low_kl = 0]
RestoreMode --> |是| LoadFullState[加载完整状态]
LoadFullState --> LoadOptimizers[加载优化器状态]
LoadFullState --> SyncLR[同步学习率]
LoadFullState --> LoadControllerState[加载控制器状态]
LoadControllerState --> PriorityCheck{存在lr_controller_state?}
PriorityCheck --> |是| LoadFromController[从controller加载]
PriorityCheck --> |否| LoadLegacy[从legacy字段加载]
LoadFromController --> Complete[完成恢复]
LoadLegacy --> Complete
ResetState --> Complete
SyncLR --> Complete
```

图表来源
- [ctde_ppo_baseline_train.py:1062-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1062-L1097)

章节来源
- [ctde_ppo_baseline_train.py:1062-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1062-L1097)

### 类图：网络与智能体关系
```mermaid
classDiagram
class ActorNetwork {
+forward(local_obs) Tensor
+get_action_probs(local_obs) Categorical
}
class CriticNetwork {
+forward(global_state) Tensor
}
class ReplayBuffer {
+store(...)
+get()
+clear()
+__len__()
}
class CTDE_PPO_Agent {
+select_actions(...)
+store_transition(...)
+compute_gae(...)
+update(force) Dict
+save(path)
+load(path, restore_training_state)
-_adapt_actor_lr_by_kl(mean_kl) str
-_update_kl_ema(mean_kl) float
-_set_actor_lr(lr) void
+_consecutive_low_kl : int
+kl_ema : float
+target_kl : float
+kl_early_stop_ratio : float
+kl_lr_low_patience : int
}
CTDE_PPO_Agent --> ActorNetwork : "使用"
CTDE_PPO_Agent --> CriticNetwork : "使用"
CTDE_PPO_Agent --> ReplayBuffer : "管理"
```

图表来源
- [ctde_ppo_baseline_train.py:460-535](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L460-L535)
- [ctde_ppo_baseline_train.py:537-566](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L537-L566)
- [ctde_ppo_baseline_train.py:779-1097](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L779-L1097)

## 依赖关系分析
- 外部依赖：torch、numpy、torch.distributions.Categorical、torch.nn.functional
- 内部依赖：FireSearchBaselineEnvironment（环境定义），信息转换模块（数据集与场景管理）
- 耦合点：
  - Agent与环境通过obs["local_obs"]、obs["global_state"]、rewards、done、info交互
  - Agent与网络通过张量形状与设备(device)一致性耦合
  - GAE与Critic共享全局状态表示，需保证维度一致

章节来源
- [ctde_ppo_baseline_train.py:30-36](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L30-L36)
- [ctde_ppo_baseline_train.py:24-28](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L24-L28)

## 性能与数值稳定性
- 数值稳定措施
  - advantages标准化：减均值、除以(std+1e-8)，避免大尺度优势导致不稳定
  - 梯度裁剪：nn.utils.clip_grad_norm_(..., max_grad_norm)防止爆炸
  - Adam eps=1e-5，避免除零
  - KL自适应学习率：多级调整策略，包含紧急下降、正常下降、耐心等待和学习率提升
- 批处理优化
  - mini_batch_size下限512，提升并行效率
  - 多epoch重复利用同一批轨迹，减少环境交互开销
  - **新增**：早停机制避免不必要的计算
- 复杂度
  - GAE：O(T)逆序扫描
  - 更新：每epoch O(B/mb_size)次前向/反向，整体O(ppo_epochs * B/mb_size)
  - **新增**：KL自适应计算为O(1)常数时间操作

章节来源
- [ctde_ppo_baseline_train.py:952](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L952)
- [ctde_ppo_baseline_train.py:981-982](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L981-L982)
- [ctde_ppo_baseline_train.py:1011-1012](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1011-L1012)
- [ctde_ppo_baseline_train.py:873-901](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L873-L901)

## 故障排查指南
- 训练不更新或更新停滞
  - 检查缓冲区长度是否达到batch_size；若不足，增大episode长度或减小batch_size
  - 观察clip_fraction是否接近1，可能说明策略更新被频繁裁剪，需降低学习率或增大epsilon
  - **新增**：检查consecutive_low_kl是否持续增长，可能需要调整kl_lr_low_patience参数
- KL发散或策略崩溃
  - 启用KL自适应学习率，适当增大kl_ema_beta或调整各级别调整因子
  - 增大max_grad_norm或降低actor_lr
  - **新增**：检查emergency_down动作频率，可能需要调整kl_lr_emergency_ratio
- 数值异常（NaN/Inf）
  - 确认advantages标准化未除零（std+1e-8）
  - 检查网络初始化与LayerNorm配置
- 评估失败或泛化差
  - 检查eval阶段是否使用确定性策略select_actions_deterministic
  - 关注validation_interval与best_val保存逻辑
- **新增**：学习率调整问题
  - 检查kl_early_stop是否频繁触发，可能需要调整kl_early_stop_ratio
  - 观察ppo_epochs_completed是否小于配置的ppo_epochs，说明早停机制在工作
  - 监控kl_lr_action的变化模式，确保学习率调整符合预期

章节来源
- [ctde_ppo_baseline_train.py:943-1060](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L943-L1060)
- [ctde_ppo_baseline_train.py:1362-1929](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1362-L1929)

## 结论
该实现采用经典PPO框架并结合多项工程优化：GAE优势估计、裁剪目标、增强的KL自适应学习率控制、梯度裁剪与advantages标准化，形成稳健的训练闭环。**新增的耐心机制和早停策略显著提升了训练的鲁棒性和效率**。ReplayBuffer以简洁的列表结构支撑整段轨迹更新，配合小批量多epoch训练，兼顾易用性与效率。通过完善的日志与质量指标，包括新的consecutive_low_kl、kl_early_stop和ppo_epochs_completed等监控指标，便于全面掌握训练稳定性与收敛性。

## 附录：超参数影响与调参建议
- clip_epsilon
  - 作用：限制ratio上下界，控制策略更新幅度
  - 影响：过小导致更新保守、收敛慢；过大易破坏旧策略，造成性能抖动
  - 建议：默认0.2，若出现大幅震荡可降至0.1~0.15；若长期不更新可增至0.25~0.3
- gae_lambda
  - 作用：GAE偏差-方差折中系数
  - 影响：接近1偏向低偏高方差；接近0偏向高偏低方差
  - 建议：默认0.95通常较稳；若回报噪声大可适当降低至0.9~0.95
- entropy_coef
  - 作用：鼓励探索的正则项权重
  - 影响：过大导致策略过于随机；过小导致过早收敛到次优
  - 建议：默认0.01；若早期探索不足可增至0.02~0.05，后期逐步衰减
- **新增**：KL自适应学习率相关参数
  - target_kl：KL散度目标值，控制策略更新的保守程度
    - 建议：默认0.0065，对于更稳定的训练可使用0.01~0.02
  - kl_lr_low_patience：耐心阈值，控制学习率提升的时机
    - 建议：默认3次，可根据训练动态调整，范围2-5
  - kl_early_stop_ratio：早停阈值，防止单个epoch过度训练
    - 建议：默认1.50，可根据任务难度调整，范围1.2-2.0
  - kl_lr_low_ratio/kl_lr_high_ratio：KL迟滞区间边界
    - 建议：默认0.80/1.20，间隔越大越保守
  - kl_lr_up_factor/kl_lr_down_factor/kl_lr_emergency_factor：各级别调整因子
    - 建议：默认1.03/0.85/0.70，根据训练稳定性微调
- 其他相关
  - max_grad_norm：防止梯度爆炸，常见0.5~1.0
  - ppo_epochs与mini_batch_size：决定数据重用与批大小，需平衡稳定性与效率
  - **新增**：ppo_epochs_completed：实际完成的epoch数，可用于监控早停效果

章节来源
- [ctde_ppo_baseline_train.py:116-128](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L116-L128)
- [ctde_ppo_baseline_train.py:788-800](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L788-L800)
- [ctde_ppo_baseline_train.py:873-901](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L873-L901)
- [ctde_ppo_baseline_train.py:1025-1032](file://environment_variables/environment_variables/ctde_ppo_baseline_train.py#L1025-L1032)