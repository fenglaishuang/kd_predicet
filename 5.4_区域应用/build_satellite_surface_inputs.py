from pathlib import Path
import numpy as np, pandas as pd, json
ROOT = Path('C:\\Users\\77940\\Documents\\Codex\\2026-09-10\\5-8-5-1-8-x20-2')
INPUT = ROOT / 'work' / 'satellite_5_4'
D = Path(__file__).resolve().parent / 'results'
D.mkdir(exist_ok=True)
kd = pd.read_csv(INPUT / 'Kd490_202507_chukchi.csv', skiprows=[1])
ch = pd.read_csv(INPUT / 'chlorophyll_202507_chukchi.csv', skiprows=[1])
df = kd.merge(ch, on=['time', 'latitude', 'longitude'], suffixes=('_kd', '_chl'))
df = df.rename(columns={'Kd_490': 'kd490', 'chlorophyll': 'chl'})
df = df[np.isfinite(df.kd490) & np.isfinite(df.chl) & (df.kd490 > 0) & (df.chl > 0)].copy()
beta = np.array([-0.18673697904279155, 0.1118165545107935])
df['S'] = np.clip(beta[0] + beta[1] * np.log(df.chl), 0.05, 8)
wl = np.linspace(410, 780, 371)
planck = wl ** (-5) / (np.exp(14387700 / (wl * 5778)) - 1)
planck /= np.trapezoid(planck, wl)
K = df.kd490.to_numpy()[:, None] * (wl[None, :] / 490.0) ** (-df.S.to_numpy()[:, None])
slow = wl <= 550
fast = wl >= 560

def eff(mask):
    return np.trapezoid(planck[mask] * K[:, mask], wl[mask], axis=1) / np.trapezoid(planck[mask], wl[mask])
df['kd_slow_surface'] = eff(slow)
df['kd_fast_surface'] = eff(fast)
doy = 197.0
decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
lat = np.deg2rad(df.latitude.to_numpy())
hour_angle = np.zeros_like(lat)
cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
df['solar_zenith_deg'] = np.rad2deg(np.arccos(np.clip(cosz, -1, 1)))
out = df[['time', 'latitude', 'longitude', 'kd490', 'chl', 'S', 'kd_slow_surface', 'kd_fast_surface', 'solar_zenith_deg']]
out.to_csv(D / 'satellite_surface_inputs_202507.csv', index=False)
meta = {'method': 'power-law Kd spectrum 410-780 nm; normalized 5778 K Planck spectrum for effective Kd weighting', 'slope_beta': beta.tolist(), 'zenith_time': '2025-07-16 local solar noon (representative monthly geometry)', 'n_valid': int(len(out)), 'bounds': {'lon': [-170, -160], 'lat': [66, 73]}}
(D / 'satellite_surface_inputs_202507_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
