import json
filepath = r'D:\DestopMoren\Desktop\Pyproject\Self-adaptive parameters\environment_variables\environment_variables\outputs\lr_comparison_20260706_164105\lr_comparison_summary.json'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

fixed_data = data['variants'].get('Fixed_LR_CTDE_PPO_seed42', {})
es = fixed_data.get('eval_summary', {}).get('best_val', {}).get('splits', {})
print('==== Fixed LR Diagnostics ====')
for split_name, split_data in es.items():
    stg = list(split_data.get('stages', {}).keys())[0] if split_data.get('stages') else 'N/A'
    metrics = split_data.get('stages', {}).get(stg, {})
    print(f'-- {split_name} --')
    print(f'Success Rate: {metrics.get("success_rate", 0)*100:.1f}%')
    print(f'Timeout Rate: {metrics.get("timeout_rate", 0)*100:.1f}%')
    print(f'Zero Cov Timeout: {metrics.get("zero_coverage_timeout_rate", 0)*100:.1f}%')
    print(f'Mean Length: {metrics.get("mean_length", 0):.1f}')
    print(f'Mean Reward: {metrics.get("mean_reward", 0):.1f}')
