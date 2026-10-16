import json
import run_ablation_nested_corrected as exp
import numpy as np
import pandas as pd

def main():
    d, _ = exp.load_data()
    exp.init_worker(d)
    original, info = exp.run_fold('K', 1)
    changed = d.copy()
    held = changed.profile_id.eq(1)
    changed.loc[held, 'kd_slow_interp'] += 7.0
    changed.loc[held, 'kd_fast_interp'] *= 11.0
    exp.init_worker(changed)
    perturbed, changed_info = exp.run_fold('K', 1)
    cols = ['pred_slow', 'pred_fast', 'baseline_slow', 'baseline_fast']
    max_diff = float(np.max(np.abs(original[cols].to_numpy() - perturbed[cols].to_numpy())))
    assert max_diff == 0 and info == changed_info
    exp.init_worker(d)
    profiles = set(d.profile_id.unique())
    missing_depth_points, fold_count = (0, 0)
    for code in exp.SETS:
        for pid in profiles:
            path = exp.OUT / 'folds' / f'{code}_{int(pid):02d}.json'
            audit = json.loads(path.read_text(encoding='utf-8'))
            assert audit['held_out_profile'] == pid and audit['combination'] == code
            validation_ids = []
            for split in audit['inner_splits']:
                tr, va = (set(split['train_profiles']), set(split['validation_profiles']))
                assert not tr & va and pid not in tr | va
                assert tr | va == profiles - {pid}
                validation_ids.extend(va)
            assert len(validation_ids) == 32 and set(validation_ids) == profiles - {pid}
            missing_depth_points += audit['outer_points_at_missing_training_depth']
            fold_count += 1
    p = pd.read_csv(exp.OUT / 'outer_predictions.csv')
    assert p.groupby('combination').size().eq(len(d)).all()
    assert p.groupby('combination').profile_id.nunique().eq(33).all()
    assert len(p) == 7 * len(d) and np.isfinite(p[cols].to_numpy()).all()
    for code in exp.SETS:
        q = p.loc[p.combination.eq(code)].merge(d, on=['profile_id', 'depth'], validate='one_to_one', suffixes=('_saved', '_source'))
        for branch in ('slow', 'fast'):
            assert np.allclose(q[f'kd_{branch}_interp_saved'], q[f'kd_{branch}_interp_source'], atol=1e-12, rtol=0)
    result = {'held_out_target_perturbation_max_prediction_difference': max_diff, 'held_out_target_perturbation_preserves_selected_parameters': info == changed_info, 'complete_outer_fold_combination_count': fold_count, 'prediction_rows': len(p), 'outer_missing_training_depth_points': missing_depth_points, 'all_checks_passed': True}
    (exp.OUT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
