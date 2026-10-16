import os
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_var] = '1'
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.runtime_pkgs_ablation'))
import argparse
import hashlib
import itertools
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.model_selection import KFold
ROOT = Path('C:/Users/77940/OneDrive/Desktop/博士毕业/外审/外审意见1/第五章加内容')
OUT = HERE / 'ablation_nested_20260917'
SETS = {'K': (0,), 'C': (1,), 'S': (2,), 'KC': (0, 1), 'KS': (0, 2), 'CS': (1, 2), 'KCS': (0, 1, 2)}
LABELS = {'K': 'Kd', 'C': 'Chl-a', 'S': 'SZA', 'KC': 'Kd + Chl-a', 'KS': 'Kd + SZA', 'CS': 'Chl-a + SZA', 'KCS': 'Kd + Chl-a + SZA'}
GRID = list(itertools.product((6, 8, 10), (30.0, 100.0, 300.0)))
SEED = 20260917
DATA = None

def metrics(y, p, ids):
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e = p - y
    per = pd.DataFrame({'id': ids, 'sq': e * e}).groupby('id').sq.mean()
    return {'n_points': len(y), 'n_profiles': len(per), 'RMSE': float(np.sqrt(np.mean(e * e))), 'MAE': float(np.mean(abs(e))), 'Bias': float(np.mean(e)), 'R2': float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)), 'mean_profile_RMSE': float(np.sqrt(per).mean())}

def load_data():
    source = ROOT / 'gam_b_depthwise_data/gam_b_depthwise_data/interp_kd_chl_density_depthwise_b_for_gam.csv'
    d = pd.read_csv(source).sort_values(['profile_id', 'depth']).reset_index(drop=True)
    assert len(d) == 3456 and d.profile_id.nunique() == 33
    assert not d.duplicated(['profile_id', 'depth']).any()
    assert d.baseline_method.eq('mean').all() and d.baseline_smooth_window.eq(1).all()
    assert d.kd_transform.eq('raw').all() and d.baseline_kd_transform.eq('raw').all()
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    for b in ('slow', 'fast'):
        path = ROOT / f'reconstruct_{b}_3layers_transition/all_profiles_reconstructed_{b}.csv'
        h = pd.read_csv(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        q = d[['profile_id', 'depth', f'kd_{b}_interp']].merge(h, on=['profile_id', 'depth'], validate='one_to_one')
        assert len(q) == len(d) and np.allclose(q[f'kd_{b}_interp'], q.kd_true, atol=1e-12, rtol=0)
        for c in ('surface_kd', 'top_chl', 'zenith_deg'):
            assert h.groupby('profile_id')[c].nunique().eq(1).all()
        meta = h.groupby('profile_id')[['surface_kd', 'top_chl', 'zenith_deg']].first()
        d[f'K_{b}'] = d.profile_id.map(meta.surface_kd)
        assert np.allclose(d.top_chl, d.profile_id.map(meta.top_chl))
        z = d.profile_id.map(meta.zenith_deg).to_numpy()
        if b == 'slow':
            d['SZA_deg'] = z
        else:
            assert np.allclose(z, d.SZA_deg)
        surface = d.groupby('profile_id')[f'kd_{b}_interp'].first()
        assert np.allclose(d[f'K_{b}'], d.profile_id.map(surface))
    d['C'] = d.top_chl
    d['S'] = np.cos(np.deg2rad(d.SZA_deg))
    cols = ['profile_id', 'depth', 'kd_slow_interp', 'kd_fast_interp', 'K_slow', 'K_fast', 'C', 'S', 'SZA_deg']
    assert np.isfinite(d[cols].to_numpy()).all()
    return (d[cols], hashes)

def init_worker(d):
    global DATA
    DATA = d

def baseline(train, branch, evaluation):
    b = train.groupby('depth')[f'kd_{branch}_interp'].mean()
    return np.interp(evaluation.depth.to_numpy(), b.index.to_numpy(), b.to_numpy())

def features(d, branch, keys, depth=False):
    x = d[[f'K_{branch}', 'C', 'S']].to_numpy()[:, keys]
    return np.column_stack([d.depth, x]) if depth else x

def predict_gam(x, y, xt, param):
    n, lam = param
    terms = s(0, n_splines=n)
    for j in range(1, x.shape[1]):
        terms += s(j, n_splines=n)
    model = LinearGAM(terms, lam=lam, max_iter=100, tol=0.0001).fit(x, y)
    if model.logs_['diffs'][-1] > model.tol:
        raise RuntimeError('GAM failed to converge')
    pred = model.predict(xt)
    assert np.isfinite(pred).all()
    return pred

def slow_parts(train, val, keys, params):
    result = []
    for lo, hi in ((0, 15), (10, 80)):
        tr = train.loc[train.depth.between(lo, hi)]
        mask = val.depth.between(lo, hi).to_numpy()
        target = tr.kd_slow_interp.to_numpy() - baseline(train, 'slow', tr)
        x = features(tr, 'slow', keys, True)
        xt = features(val.loc[mask], 'slow', keys, True)
        output = np.zeros((len(params), len(val)))
        for j, par in enumerate(params):
            if mask.any():
                output[j, mask] = predict_gam(x, target, xt, par)
        result.append(output)
    return result

def blend(depth, shallow, deep):
    w = 0.5 * (1 + np.cos(np.pi * np.clip((depth - 10) / 5, 0, 1)))
    return shallow * w + deep * (1 - w)

def fast_targets(train):
    b = baseline(train, 'fast', train)
    rows = []
    for pid, g in train.groupby('profile_id', sort=True):
        mask = train.profile_id.eq(pid).to_numpy()
        a, c = np.linalg.lstsq(np.column_stack([b[mask], np.ones(mask.sum())]), g.kd_fast_interp.to_numpy(), rcond=None)[0]
        first = g.iloc[0]
        rows.append({'profile_id': pid, 'a': a, 'c': c, 'K_fast': first.K_fast, 'C': first.C, 'S': first.S})
    return pd.DataFrame(rows)

def fast_candidates(train, val, keys, params):
    targets = fast_targets(train)
    vm = val.groupby('profile_id', sort=True).first().reset_index()
    x = features(targets, 'fast', keys)
    xt = features(vm, 'fast', keys)
    b = baseline(train, 'fast', val)
    outputs = []
    for par in params:
        a = np.clip(predict_gam(x, targets.a.to_numpy(), xt, par), 0.2, 3.0)
        c = predict_gam(x, targets.c.to_numpy(), xt, par)
        ia = val.profile_id.map(dict(zip(vm.profile_id, a))).to_numpy()
        ic = val.profile_id.map(dict(zip(vm.profile_id, c))).to_numpy()
        outputs.append(ia * b + ic)
    return np.asarray(outputs)

def score(y, p, ids):
    sq = (p - y) ** 2
    means = [np.sqrt(np.mean(sq[ids == pid])) for pid in np.unique(ids)]
    return (float(np.mean(means)), float(np.sqrt(np.mean(sq))))

def run_fold(code, held):
    d = DATA
    train = d.loc[d.profile_id.ne(held)].reset_index(drop=True)
    test = d.loc[d.profile_id.eq(held)].reset_index(drop=True)
    ids = np.sort(train.profile_id.unique())
    assert held not in ids
    keys = SETS[code]
    ps = np.full((len(GRID), len(train)), np.nan)
    pdp = np.full_like(ps, np.nan)
    pf = np.full_like(ps, np.nan)
    bs = np.full(len(train), np.nan)
    split_audit = []
    splitter = KFold(n_splits=4, shuffle=True, random_state=SEED)
    for tri, vai in splitter.split(ids):
        train_ids, val_ids = (ids[tri], ids[vai])
        assert not set(train_ids) & set(val_ids)
        tr = train.loc[train.profile_id.isin(train_ids)].reset_index(drop=True)
        mask = train.profile_id.isin(val_ids).to_numpy()
        val = train.loc[mask].reset_index(drop=True)
        bs[mask] = baseline(tr, 'slow', val)
        a, b = slow_parts(tr, val, keys, GRID)
        ps[:, mask], pdp[:, mask] = (a, b)
        pf[:, mask] = fast_candidates(tr, val, keys, GRID)
        split_audit.append({'train_profiles': train_ids.tolist(), 'validation_profiles': val_ids.tolist(), 'validation_points_at_missing_training_depth': int((~val.depth.isin(tr.depth)).sum())})
    assert all((np.isfinite(a).all() for a in (ps, pdp, pf, bs)))
    y, yi, depths = (train.kd_slow_interp.to_numpy(), train.profile_id.to_numpy(), train.depth.to_numpy())
    candidates = []
    for j in range(len(GRID)):
        for k in range(len(GRID)):
            p = bs + blend(depths, ps[j], pdp[k])
            candidates.append((*score(y, p, yi), j, k))
    bestslow = min(candidates)
    bestfast = min(((*score(train.kd_fast_interp.to_numpy(), p, yi), j) for j, p in enumerate(pf)))
    j, k = bestslow[-2:]
    shallow = slow_parts(train, test, keys, [GRID[j]])[0][0]
    deep = slow_parts(train, test, keys, [GRID[k]])[1][0]
    pred_slow = baseline(train, 'slow', test) + blend(test.depth.to_numpy(), shallow, deep)
    pred_fast = fast_candidates(train, test, keys, [GRID[bestfast[-1]]])[0]
    out = test[['profile_id', 'depth', 'kd_slow_interp', 'kd_fast_interp']].copy()
    out['combination'] = code
    out['pred_slow'], out['pred_fast'] = (pred_slow, pred_fast)
    out['baseline_slow'] = baseline(train, 'slow', test)
    out['baseline_fast'] = baseline(train, 'fast', test)
    info = {'combination': code, 'held_out_profile': int(held), 'inner_splits': split_audit, 'slow_shallow': GRID[j], 'slow_deep': GRID[k], 'fast': GRID[bestfast[-1]], 'slow_inner_mean_profile_RMSE': bestslow[0], 'fast_inner_mean_profile_RMSE': bestfast[0], 'outer_points_at_missing_training_depth': int((~test.depth.isin(train.depth)).sum())}
    return (out, info)

def collect(d):
    frames = [pd.read_csv(OUT / 'folds' / f'{code}_{int(pid):02d}.csv') for code in SETS for pid in sorted(d.profile_id.unique())]
    pred = pd.concat(frames, ignore_index=True)
    assert len(pred) == len(d) * len(SETS)
    assert not pred.duplicated(['combination', 'profile_id', 'depth']).any()
    pred.to_csv(OUT / 'outer_predictions.csv', index=False)
    summary, per = ([], [])
    panels = [('Slow', '0-15 m', 'slow', lambda x: x.depth.lt(15)), ('Slow', '15-80 m', 'slow', lambda x: x.depth.ge(15)), ('Fast', '0-80 m', 'fast', lambda x: x.depth.between(0, 80))]
    for comp, seg, branch, maskfun in panels:
        for code in SETS:
            q = pred.loc[pred.combination.eq(code) & maskfun(pred)]
            m = metrics(q[f'kd_{branch}_interp'].to_numpy(), q[f'pred_{branch}'].to_numpy(), q.profile_id.to_numpy())
            summary.append({'component': comp, 'segment': seg, 'combination': code, 'inputs': LABELS[code], **m})
            for pid, g in q.groupby('profile_id'):
                err = g[f'pred_{branch}'] - g[f'kd_{branch}_interp']
                per.append({'component': comp, 'segment': seg, 'combination': code, 'profile_id': pid, 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'MAE': float(np.mean(abs(err)))})
    summary = pd.DataFrame(summary)
    profile = pd.DataFrame(per)
    summary.to_csv(OUT / 'summary.csv', index=False)
    profile.to_csv(OUT / 'profile_metrics.csv', index=False)
    paired = []
    rng = np.random.default_rng(SEED)
    for (comp, seg), g in profile.groupby(['component', 'segment'], sort=False):
        w = g.pivot(index='profile_id', columns='combination', values='RMSE')
        samples = rng.integers(0, len(w), size=(5000, len(w)))
        for code in SETS:
            delta = (w[code] - w.KCS).to_numpy()
            interval = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
            paired.append({'component': comp, 'segment': seg, 'combination': code, 'mean_profile_RMSE_difference_vs_KCS': float(delta.mean()), 'bootstrap_95_low': interval[0], 'bootstrap_95_high': interval[1], 'profiles_better_than_KCS': int((delta < 0).sum())})
    pd.DataFrame(paired).to_csv(OUT / 'paired_profile_differences.csv', index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--collect-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / 'folds').mkdir(exist_ok=True)
    d, hashes = load_data()
    manifest = {'input_sha256': hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'seed': SEED, 'grid': GRID, 'inputs': LABELS, 'outer': 'Leave one profile out', 'inner': '4-fold shuffled profiles', 'selection_metric': 'Mean profile RMSE; point RMSE breaks ties', 'slow_selection': 'Joint shallow/deep hyperparameters scored after blending', 'baseline': 'Training-fold depth mean; training-only linear interpolation; nearest boundary if unsupported', 'interactions': 'None in any combination', 'radar_subset': ['KCS', 'CS', 'KS', 'KC']}
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous != json.loads(json.dumps(manifest)) and (not args.collect_only):
            raise RuntimeError('Existing manifest differs; use a separate output folder.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.collect_only:
        jobs = [(c, int(p)) for c in SETS for p in sorted(d.profile_id.unique()) if not (OUT / 'folds' / f'{c}_{int(p):02d}.json').exists()]
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(d,)) as pool:
            futures = {pool.submit(run_fold, c, p): (c, p) for c, p in jobs}
            for done, fut in enumerate(as_completed(futures), 1):
                c, p = futures[fut]
                result, info = fut.result()
                result.to_csv(OUT / 'folds' / f'{c}_{p:02d}.csv', index=False)
                (OUT / 'folds' / f'{c}_{p:02d}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    collect(d)
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
if __name__ == '__main__':
    main()
