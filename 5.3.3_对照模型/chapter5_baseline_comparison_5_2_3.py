import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DEPTH_BINS = [(0, 15), (15, 30), (30, 50), (50, 80)]

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {'n_points': 0, 'RMSE': np.nan, 'MAE': np.nan, 'Bias': np.nan, 'R2': np.nan, 'MedianAE': np.nan, 'MAPE': np.nan}
    y_t = y_true[mask]
    y_p = y_pred[mask]
    err = y_p - y_t
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    bias = float(np.mean(err))
    median_ae = float(np.median(np.abs(err)))
    r2 = float(r2_score(y_t, y_p)) if y_t.size > 1 else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.nanmean(np.abs(err / y_t) * 100.0)) if np.any(y_t != 0) else np.nan
    return {'n_points': int(mask.sum()), 'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'R2': r2, 'MedianAE': median_ae, 'MAPE': mape}

def loocv_profile_linear(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> np.ndarray:
    profile_ids = df['profile_id'].to_numpy(dtype=int)
    preds = np.full(len(df), np.nan, dtype=float)
    unique_profiles = np.unique(profile_ids)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    if len(feature_cols) == 0:
        fill_val = np.nanmean(y)
        preds[:] = fill_val
        return preds
    for pid in unique_profiles:
        test_mask = profile_ids == pid
        train_mask = ~test_mask
        x_train = X[train_mask]
        y_train = y[train_mask]
        x_test = X[test_mask]
        if x_train.shape[0] < 2 or not np.any(train_mask):
            if x_train.size == 0:
                pred_val = float(np.nanmean(y_train if y_train.size else y))
                preds[test_mask] = pred_val
            else:
                model = LinearRegression(fit_intercept=True)
                model.fit(x_train, y_train)
                preds[test_mask] = model.predict(x_test)
            continue
        train_valid = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
        x_train = x_train[train_valid]
        y_train = y_train[train_valid]
        if x_train.size == 0:
            preds[test_mask] = float(np.nanmean(y))
            continue
        test_valid = np.isfinite(x_test).all(axis=1)
        if not np.any(test_valid):
            preds[test_mask] = np.nan
            continue
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        preds[test_mask] = model.predict(x_test)
    return preds

def build_model_predictions(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == 'slow':
        top_kd_col = 'top_kd_slow'
    elif mode == 'fast':
        top_kd_col = 'top_kd_fast'
    else:
        raise ValueError('mode must be slow or fast')
    require_cols = {'profile_id', 'depth', 'kd_true', 'baseline_y', top_kd_col}
    if not require_cols.issubset(df.columns):
        miss = require_cols - set(df.columns)
        raise ValueError(f'缺少必要列: {miss}')
    data = df[['profile_id', 'depth', 'kd_true', 'baseline_y', top_kd_col, 'top_chl', 'zenith_deg', 'kd_reconstructed']].copy()
    for c in ['top_kd_slow', 'top_kd_fast', 'top_chl', 'zenith_deg', 'baseline_y', 'kd_reconstructed', 'kd_true', top_kd_col]:
        if c in data.columns:
            if not pd.api.types.is_numeric_dtype(data[c]):
                data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(subset=['kd_true', 'baseline_y', top_kd_col])
    data = data.reset_index(drop=True)
    data['pred_baseline_depthwise'] = data['baseline_y'].to_numpy(dtype=float)
    X_top = data[[top_kd_col]].copy()
    data['pred_surface_kd'] = loocv_profile_linear(data, [top_kd_col], 'kd_true')
    feat_full = [top_kd_col]
    if 'top_chl' in data.columns and data['top_chl'].notna().any():
        feat_full.append('top_chl')
    if 'zenith_deg' in data.columns and data['zenith_deg'].notna().any():
        feat_full.append('zenith_deg')
    data_lin = data.dropna(subset=feat_full)
    data.loc[:, 'pred_linear_topchlzenith'] = np.nan
    data.loc[data_lin.index, 'pred_linear_topchlzenith'] = loocv_profile_linear(data.loc[data_lin.index].reset_index(drop=True), feat_full, 'kd_true')
    data['pred_gam'] = data['kd_reconstructed'].to_numpy(dtype=float)
    return data

def depth_bin_rows(df: pd.DataFrame, pred_col: str, true_col: str, out_prefix: str) -> List[Dict[str, float]]:
    rows = []
    for low, high in DEPTH_BINS:
        sub = df[(df['depth'] >= low) & (df['depth'] < high)]
        m = calc_metrics(sub[true_col].to_numpy(dtype=float), sub[pred_col].to_numpy(dtype=float))
        m['depth_bin'] = f'{low}-{high}m'
        m['model'] = out_prefix
        rows.append(m)
    return rows

def profile_rows(df: pd.DataFrame, pred_cols: List[str], true_col: str) -> List[Dict[str, float]]:
    out = []
    for pid, g in df.groupby('profile_id'):
        row = {'profile_id': int(pid), 'n_points': int(len(g))}
        y_true = g[true_col].to_numpy(dtype=float)
        for pcol in pred_cols:
            metrics = calc_metrics(y_true, g[pcol].to_numpy(dtype=float))
            row[f'{pcol}_rmse'] = metrics['RMSE']
            row[f'{pcol}_mae'] = metrics['MAE']
            row[f'{pcol}_r2'] = metrics['R2']
            row[f'{pcol}_bias'] = metrics['Bias']
        out.append(row)
    return out

def summarize_mode(df: pd.DataFrame, mode: str, out_dir: Path) -> None:
    true_col = 'kd_true'
    model_cols = ['pred_baseline_depthwise', 'pred_surface_kd', 'pred_linear_topchlzenith', 'pred_gam']
    y_true = df[true_col].to_numpy(dtype=float)
    summary_rows = []
    for pcol in model_cols:
        m = calc_metrics(y_true, df[pcol].to_numpy(dtype=float))
        m['model'] = pcol
        summary_rows.append(m)
    base = df[['pred_baseline_depthwise', true_col]].copy().dropna()
    base_rmse = calc_metrics(base[true_col].to_numpy(float), base['pred_baseline_depthwise'].to_numpy(float))['RMSE']
    for row in summary_rows:
        row['improve_rmse_vs_depthwise_pct'] = (base_rmse - row['RMSE']) / base_rmse * 100.0 if np.isfinite(base_rmse) and base_rmse > 0 else np.nan
    depth_rows = []
    for pcol in model_cols:
        depth_rows.extend(depth_bin_rows(df, pcol, true_col, pcol))
    prof_rows = profile_rows(df, model_cols, true_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_csv = out_dir / f'baseline_summary_{mode}.csv'
    depth_csv = out_dir / f'baseline_depth_bins_{mode}.csv'
    profile_csv = out_dir / f'baseline_profile_metrics_{mode}.csv'
    pred_csv = out_dir / f'baseline_model_predictions_{mode}.csv'
    pd.DataFrame(summary_rows).to_csv(global_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(depth_rows).to_csv(depth_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(prof_rows).to_csv(profile_csv, index=False, encoding='utf-8-sig')
    pred_keep = ['profile_id', 'depth', 'kd_true', 'baseline_y', 'top_kd_slow', 'top_kd_fast', 'top_chl', 'zenith_deg']
    pred_keep = [c for c in pred_keep if c in df.columns]
    pred_keep.extend(model_cols)
    df[pred_keep].to_csv(pred_csv, index=False, encoding='utf-8-sig')
    sm = pd.DataFrame(summary_rows)
    ordered = ['pred_baseline_depthwise', 'pred_surface_kd', 'pred_linear_topchlzenith', 'pred_gam']
    sm = sm.set_index('model').loc[ordered].reset_index()
    pretty = pd.DataFrame({'model': ['Baseline A(Depthwise baseline)', 'Baseline B(Top-Kd linear)', 'Baseline C(Linear regression)', 'Proposed(GAM)'], 'RMSE': sm['RMSE'].to_numpy(), 'MAE': sm['MAE'].to_numpy(), 'R2': sm['R2'].to_numpy(), 'Bias': sm['Bias'].to_numpy(), 'MedianAE': sm['MedianAE'].to_numpy(), 'MAPE': sm['MAPE'].to_numpy(), 'Improve_vs_depthwise(%)': sm['improve_rmse_vs_depthwise_pct'].to_numpy()})
    pretty.to_csv(out_dir / f'baseline_chapter5_5.2.3_table_{mode}.csv', index=False, encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='C:/Users/77940/OneDrive/Desktop/博士毕业/外审/外审意见1/第五章加内容')
    parser.add_argument('--mode', nargs='+', default=['slow', 'fast'], choices=['slow', 'fast', 'both'])
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'analysis_outputs' / 'baseline_5_2_3'))
    args = parser.parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    modes = ['slow', 'fast'] if 'both' in args.mode else args.mode
    for mode in modes:
        src = base / f'reconstruct_{mode}_3layers_transition' / f'all_profiles_reconstructed_{mode}.csv'
        if not src.exists():
            raise FileNotFoundError(f'文件不存在: {src}')
        df = pd.read_csv(src)
        pred_df = build_model_predictions(df, mode)
        summarize_mode(pred_df, mode, out_dir)
if __name__ == '__main__':
    main()
