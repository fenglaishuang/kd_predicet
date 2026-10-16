from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.runtime_pkgs_ablation'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT = Path('C:/Users/77940/OneDrive/Desktop/博士毕业/外审/外审意见1/第五章加内容')
OUT = HERE / 'section_5_3_5_outputs'
OUT.mkdir(exist_ok=True)
records, layers, bins, mats, all_data = ([], [], [], {}, {})
for branch in ['Slow', 'Fast']:
    path = ROOT / f'reconstruct_{branch.lower()}_3layers_transition' / f'all_profiles_reconstructed_{branch.lower()}.csv'
    df = pd.read_csv(path)
    truth = 'kd_true'
    df = df[['profile_id', 'depth', truth, 'kd_reconstructed']].rename(columns={truth: 'observed', 'kd_reconstructed': 'predicted'})
    assert not df.duplicated(['profile_id', 'depth']).any()
    assert np.isfinite(df.to_numpy()).all()
    assert df.profile_id.nunique() == 33 and len(df) == 3456
    df['error'] = df.predicted - df.observed
    all_data[branch] = df
    for left, right in [(0, 15), (15, 30), (30, 50), (50, 80)]:
        sub = df[(df.depth >= left) & (df.depth < right if right < 80 else df.depth <= right)]
        layers.append(dict(component=branch, depth_interval=f'{left}-{right}', points=len(sub), profiles=sub.profile_id.nunique(), RMSE=np.sqrt(np.mean(sub.error ** 2)), MAE=np.mean(abs(sub.error)), Bias=sub.error.mean()))
    for left in range(0, 80, 5):
        sub = df[(df.depth >= left) & (df.depth < left + 5 if left < 75 else df.depth <= 80)]
        bins.append(dict(component=branch, depth=left + 2.5, RMSE=np.sqrt(np.mean(sub.error ** 2)), MAE=np.mean(abs(sub.error)), Bias=sub.error.mean(), profiles=sub.profile_id.nunique()))
    mats[branch] = df.pivot(index='depth', columns='profile_id', values='error').reindex(index=np.arange(0.5, 80.5, 0.5), columns=range(1, 34))
    for pid, sub in df.groupby('profile_id'):
        sub = sub.sort_values('depth')
        grid = np.arange(16, 81, 2)
        grid = grid[(grid >= sub.depth.min()) & (grid <= sub.depth.max())]
        obs = np.interp(grid, sub.depth, sub.observed)
        pred = np.interp(grid, sub.depth, sub.predicted)
        span = grid[-1] - grid[0] if len(grid) > 1 else np.nan
        roughness = np.mean(np.abs(np.gradient(obs, grid))) if len(grid) > 2 else np.nan
        records.append(dict(component=branch, profile_id=int(pid), n_grid=len(grid), depth_span=span, vertical_variation=roughness, grid_RMSE=np.sqrt(np.mean((pred - obs) ** 2)), profile_RMSE=np.sqrt(np.mean(sub.error ** 2))))
pr = pd.DataFrame(records)
dep = pd.DataFrame(bins)
layer = pd.DataFrame(layers)
pr.to_csv(OUT / 'profile_structure_metrics.csv', index=False)
dep.to_csv(OUT / 'depth_5m_metrics.csv', index=False)
layer.to_csv(OUT / 'depth_interval_metrics.csv', index=False)
stats = []
for branch in ['Slow', 'Fast']:
    sub = pr[pr.component == branch]
    result = spearmanr(sub.vertical_variation, sub.grid_RMSE)
    stats.append(dict(component=branch, spearman_rho=float(result.statistic), p_value=float(result.pvalue), n=len(sub)))
pd.DataFrame(stats).to_csv(OUT / 'structure_associations.csv', index=False)

