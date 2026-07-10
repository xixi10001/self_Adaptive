---
kind: design
name: near_prob 退火与 target 进度绑定且终末专注阶段强制评估条件
source: session
category: adr
---

# near_prob 退火与 target 进度绑定且终末专注阶段强制评估条件

_来源：747b01c → 2ba7d62 提交周期内记录的编码计划——内容为规划时意图，实现可能滞后或有出入。_

**状态：** accepted

## 背景
课程机制中 near_prob（近起点辅助概率）与 target（覆盖要求目标）原本独立推进，导致可能出现 target=0.35 但 near_prob 已降至 0.0 的错配；同时训练尾部缺乏在最终评估条件（target=0.60, near_prob=0.0）下充分适应的机制，且早期 stage3 内（target=0.20/0.35）的验证结果被用于 best_val 选择，与真实评估条件不一致。

## 决策驱动
- 课程难度与辅助强度同步
- 最终模型在评估条件下具备足够训练量
- best_val 选择公平性

## 备选方案
- **保持 near_prob 与 target 独立推进 + 无终末专注** _（已否决）_ — 优点：改动最小，不改变现有节奏；缺点：允许 target 低而 near_prob 为 0 的错配；尾部可能未充分适应 0.60 条件；early-stage 验证误导 best_val
- **near_prob 索引不超过 target 索引 + 最后 300 回合强制评估条件** — 优点：保证辅助强度随难度递增而递减；确保尾部在评估条件下训练；is_best_val 仅在终末阶段生效避免误选；缺点：轻微改变训练节奏；需维护 _terminal_focus_active 标记和日志字段

## 决策
在 CurriculumManager._try_advance_near_prob() 中添加 near_idx >= _s3_target_idx 的硬约束，使 near_prob 退火不超前于 target 推进；新增 TERMINAL_FOCUS_EPISODES=300 常量及 activate_terminal_focus() 方法，在训练最后 300 回合将 target 和 near_prob 同时推至最终值；is_best_val 比较仅在该阶段激活时生效。

## 影响
训练节奏在正常推进阶段不受影响（约束仅在 near_idx > target_idx 时触发），尾部 300 回合专注于评估条件，best_val 选择更可靠。回退方案：将 TERMINAL_FOCUS_EPISODES 设为 0 禁用终末专注，或删除联动约束的一行 return 恢复独立推进。