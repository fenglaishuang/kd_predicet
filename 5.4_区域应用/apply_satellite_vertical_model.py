from pathlib import Path
import numpy as np, pandas as pd, json
from pygam import LinearGAM, s, te
ROOT = Path('C:\\Users\\77940\\Documents\\Codex\\2026-09-10\\5-8-5-1-8-x20-2')
P = Path('C:\\Users\\77940\\OneDrive\\Desktop\\博士毕业\\外审\\外审意见1\\第五章加内容')
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
sat = pd.read_csv(OUT / 'satellite_surface_inputs_202507.csv')
slow = pd.read_csv(P / 'reconstruct_slow_3layers_transition' / 'all_profiles_reconstructed_slow.csv')
fast = pd.read_csv(P / 'reconstruct_fast_3layers_transition' / 'all_profiles_reconstructed_fast.csv')
fp = pd.read_csv(P / 'reconstruct_fast_3layers_transition' / 'profile_predictions_fast.csv')
zgrid = np.array([2.5, 5, 7.5, 10.5, 13.5, 17.5, 21.5, 26.5, 31.5, 37.5, 43.5, 50.5, 57.5, 65.5, 73.5, 80.0])

def baseline(frame, col):
    g = frame.groupby('depth')[col].mean().dropna()
    return np.interp(zgrid, g.index, g.values)
Bslow = baseline(slow, 'B_y_slow_raw')
Bfast = baseline(fast, 'B_y_fast_raw')

def slow_X(d, powv):
    return np.column_stack([np.power(d.top_chl.to_numpy(float), powv), d.surface_kd.to_numpy(float), d.depth.to_numpy(float)])
pred_s = np.empty((len(sat), len(zgrid)))
pred_s[:] = np.nan
for lo, hi, powv in [(0, 15, 0.8), (10, 80, 0.9)]:
    d = slow[(slow.depth >= lo) & (slow.depth <= hi)].dropna(subset=['b_slow_depthwise', 'top_chl', 'surface_kd', 'depth'])
    model = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=10), lam=30).fit(slow_X(d, powv), d.b_slow_depthwise.to_numpy(float))
    for j, z in enumerate(zgrid):
        if lo <= z <= hi:
            X = np.column_stack([np.power(sat.chl.to_numpy(float), powv), sat.kd_slow_surface.to_numpy(float), np.full(len(sat), z)])
            pred_s[:, j] = Bslow[j] + model.predict(X)
prof = fp.dropna(subset=['surface_kd', 'zenith_deg', 'pred_a_depth_0_80m', 'pred_c_depth_0_80m']).copy()
groups = prof.profile_id.to_numpy()
X = np.column_stack([prof.surface_kd.to_numpy(float), np.cos(np.deg2rad(prof.zenith_deg.to_numpy(float)))])

def lopo(target):
    out = np.zeros(len(prof))
    for g in np.unique(groups):
        tr = groups != g
        tei = groups == g
        out[tei] = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0, 1, n_splines=[8, 8]), lam=30).fit(X[tr], target[tr]).predict(X[tei])
    return out
a_pred = lopo(prof.pred_a_depth_0_80m.to_numpy(float))
c_pred = lopo(prof.pred_c_depth_0_80m.to_numpy(float))
fast_score = float(np.sqrt(np.mean((a_pred - prof.pred_a_depth_0_80m) ** 2) + np.mean((c_pred - prof.pred_c_depth_0_80m) ** 2)))
model_a = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0, 1, n_splines=[8, 8]), lam=30).fit(X, prof.pred_a_depth_0_80m)
model_c = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0, 1, n_splines=[8, 8]), lam=30).fit(X, prof.pred_c_depth_0_80m)
Xa = np.column_stack([sat.kd_fast_surface.to_numpy(float), np.cos(np.deg2rad(sat.solar_zenith_deg.to_numpy(float)))])
a = np.clip(model_a.predict(Xa), 0.2, 3)
c = model_c.predict(Xa)
pred_f = np.array([a[:, None] * Bfast[None, :] + c[:, None] for _ in [0]])[0]
records = []
for comp, arr in [('slow', pred_s), ('fast', pred_f)]:
    for j, z in enumerate(zgrid):
        q = sat[['time', 'latitude', 'longitude', 'kd490', 'chl', 'solar_zenith_deg']].copy()
        q['component'] = comp
        q['depth_m'] = z
        q['kd_reconstructed'] = arr[:, j]
        records.append(q)
out = pd.concat(records, ignore_index=True)
out['kd_reconstructed'] = out.kd_reconstructed.clip(lower=0.001)
out.to_csv(OUT / 'satellite_vertical_kd_202507.csv', index=False)
summary = {'n_pixels': int(len(sat)), 'depths_m': zgrid.tolist(), 'slow_config': {'features': 'chl^0.80/0.90 + surface kd + depth', 'splines': 10, 'lam': 30}, 'fast_config': {'features': 'surface kd + cos(zenith), a/c GAM', 'splines': 8, 'lam': 30, 'LOPO_match_rmse': fast_score}, 'slow_baseline': Bslow.tolist(), 'fast_baseline': Bfast.tolist()}
(OUT / 'satellite_vertical_model_metadata.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
