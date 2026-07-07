import json
filepath = r'D:\DestopMoren\Desktop\Pyproject\Self-adaptive parameters\environment_variables\environment_variables\outputs\lr_comparison_20260706_164105\lr_comparison_summary.json'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

for v_name, v_data in data['variants'].items():
    print(f'==== {v_name} ====')
    print(f'Final Stage: {v_data.get("final_stage")} | Last Task Score: {v_data.get("last_task_score"):.3f} | Last Cov: {v_data.get("last_coverage"):.3f}')
    
    es = v_data.get('eval_summary', {}).get('best_val', {}).get('splits', {})
    if not es:
        continue
    for split_name, split_data in es.items():
        stg = list(split_data.get('stages', {}).keys())[0] if split_data.get('stages') else 'N/A'
        metrics = split_data.get('stages', {}).get(stg, {})
        print(f'  {split_name} (Stage {stg}): SR={metrics.get("success_rate",0)*100:.1f}% | Cov={metrics.get("mean_coverage",0)*100:.1f}% | Rew={metrics.get("mean_reward",0):.1f}')
    print()
