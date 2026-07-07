import json
import numpy as np

filepath = r'D:\DestopMoren\Desktop\Pyproject\Self-adaptive parameters\environment_variables\environment_variables\outputs\lr_comparison_20260706_164105\lr_comparison_summary.json'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fixed LR in detail
fixed = data['variants']['Fixed_LR_CTDE_PPO_seed42']

# quality metrics
qm = fixed.get('quality_metrics', {})
ce = qm.get('convergence_efficiency', {})
rs = qm.get('reward_stability', {})
ks = qm.get('kl_stability', {})

print('== Convergence Efficiency ==')
for k, v in ce.items():
    print(f'  {k}: {v}')

print('== Reward Stability ==')
for k, v in rs.items():
    print(f'  {k}: {v}')

print('== KL Stability ==')
for k, v in ks.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.5f}')
    else:
        print(f'  {k}: {v}')

print()
# Consult console log for stage transition info
log_path = r'D:\DestopMoren\Desktop\Pyproject\Self-adaptive parameters\environment_variables\environment_variables\outputs\lr_comparison_20260706_164105\ÑµÁ·½á¹û\Fixed_LR_CTDE_PPO_seed42\train_console_log.txt'
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total console log lines: {len(lines)}')

# Find stage transitions
for i, line in enumerate(lines):
    if '¿Î³Ì½×¶Î' in line and '->' in line:
        print(f'L{i+1}: {line.rstrip()}')

# Find init_area_percent changes
for i, line in enumerate(lines):
    if 'init_area_percent' in line and 'curriculum' in line:
        print(f'L{i+1}: {line.rstrip()}')

# Print last 20 lines
print('\n== Last 20 log lines ==')
for line in lines[-20:]:
    print(line.rstrip())
