from pathlib import Path
import numpy as np, pandas as pd, json
ROOT = Path('C:\\Users\\77940\\Documents\\Codex\\2026-09-10\\5-8-5-1-8-x20-2')
O = Path(__file__).resolve().parent / 'results'
P = Path('C:\\Users\\77940\\OneDrive\\Desktop\\博士毕业\\外审\\外审意见1\\第五章加内容')
sat = pd.read_csv(O / 'satellite_surface_inputs_202507_calibrated.csv')
fp = pd.read_csv(P / 'reconstruct_fast_3layers_transition' / 'profile_predictions_fast.csv')
base = pd.read_csv(P / 'reconstruct_fast_3layers_transition' / 'all_profiles_reconstructed_fast.csv')
z = np.array([2.5, 5, 7.5, 10.5, 13.5, 17.5, 21.5, 26.5, 31.5, 37.5, 43.5, 50.5, 57.5, 65.5, 73.5, 80.0])
B = base.groupby('depth').B_y_fast_raw.mean().dropna()
B = np.interp(z, B.index, B.values)
from pygam import LinearGAM, s, te
fp = fp.dropna(subset=['surface_kd', 'zenith_deg', 'pred_a_depth_0_80m', 'pred_c_depth_0_80m'])
X = np.column_stack([fp.surface_kd.to_numpy(float), np.cos(np.deg2rad(fp.zenith_deg.to_numpy(float)))])
Xa = np.column_stack([sat.kd_fast_calibrated.to_numpy(float), np.cos(np.deg2rad(sat.solar_zenith_deg.to_numpy(float)))])
model_a = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0, 1, n_splines=[8, 8]), lam=30).fit(X, fp.pred_a_depth_0_80m.to_numpy(float))
model_c = LinearGAM(s(0, n_splines=8) + s(1, n_splines=8) + te(0, 1, n_splines=[8, 8]), lam=30).fit(X, fp.pred_c_depth_0_80m.to_numpy(float))
a = np.clip(model_a.predict(Xa), 0.2, 3)
c = model_c.predict(Xa)
arr = np.maximum(a[:, None] * B[None, :] + c[:, None], 0.001)
assert arr.shape == (len(sat), len(z)) and np.isfinite(arr).all()
full = pd.read_csv(O / 'satellite_vertical_kd_202507.csv')
full = full[full.component != 'fast']
rec = []
for j, zz in enumerate(z):
    q = sat[['time', 'latitude', 'longitude', 'kd490', 'chl', 'solar_zenith_deg']].copy()
    q['component'] = 'fast'
    q['depth_m'] = zz
    q['kd_reconstructed'] = arr[:, j]
    rec.append(q)
full = pd.concat([full, pd.concat(rec)], ignore_index=True)
full.to_csv(O / 'satellite_vertical_kd_202507_calibrated_fast.csv', index=False)
