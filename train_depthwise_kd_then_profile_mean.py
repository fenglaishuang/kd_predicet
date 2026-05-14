# -*- coding: utf-8 -*-
"""
train_depthwise_kd_error_first.py

核心改进：
1. 仍然使用逐深度点训练，不再先对 profile 求平均；
2. 默认使用 Leave-One-Profile-Out CV，避免同一剖面的深度点泄漏；
3. 模型选择不以相关系数 r 为主，而以预测值与真实值的误差为主；
4. 默认模型选择指标为 profile_rmse_mean：
      先计算每条 profile 的逐点 KD RMSE，再对所有 profile 求平均；
5. 同时输出：
      - 全局逐点误差：point_kd_rmse / point_kd_mae / bias / medianAE / q90AE
      - 剖面逐点误差：profile_rmse_mean / profile_mae_mean / profile_rmse_median
      - 剖面平均值误差：profile_mean_kd_rmse
      - 相关系数 r 仅作为辅助指标；
6. 支持：
      - depthwise_kd
      - depthwise_residual
      - slow / fast
      - 分深度段训练
      - GAM 网格搜索
      - 保存最佳模型
      - 保存最佳 CV 预测
      - 保存 final-fit 预测
      - 保存逐剖面重建图、误差直方图、误差随深度变化图、profile RMSE 排序图
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.io

from pygam import LinearGAM, s, te
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# =========================================================
# 1. 路径设置
# =========================================================

input_csv = r"D:\work\gam_b_depthwise_data\interp_kd_chl_density_depthwise_b_for_gam.csv"
zenith_mat_path = r"D:\work\zenith.mat"

# "slow" 或 "fast"
target_branch = "slow"

# "depthwise_kd"       -> 直接训练逐深度 KD
# "depthwise_residual" -> 训练逐深度 residual，重建时 pred_kd = baseline + pred_residual
training_target_mode = "depthwise_residual"

output_dir = rf"D:\work\gam_depthwise_training_{training_target_mode}_{target_branch}_error_first"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 2. 基础列名
# =========================================================

profile_col = "profile_id"
depth_col = "depth"
chl_col = "top_chl"

if target_branch == "slow":
    kd_col = "kd_slow_interp"
    y_col = "Y_slow_interp"
    baseline_col = "B_y_slow"
    residual_col = "b_slow_depthwise"
elif target_branch == "fast":
    kd_col = "kd_fast_interp"
    y_col = "Y_fast_interp"
    baseline_col = "B_y_fast"
    residual_col = "b_fast_depthwise"
else:
    raise ValueError("target_branch 只能是 'slow' 或 'fast'")


# =========================================================
# 3. 可调参数
# =========================================================

exclude_profile_ids = []

# 分段训练。
# 注意：默认前一段为 [zmin, zmax)，最后一段为 [zmin, zmax]，避免边界深度重复进入两个模型。
depth_ranges = [
    (0, 15),
    (10, 80),
]
last_range_include_right = True

min_points_per_profile_in_range = 3
profile_id_order_in_mat = None

# 逐深度训练建议包含 depth 或 baseline，否则模型容易在同一 profile 内预测成常数。
model_mode_candidates = [
    "depth_only",
    # "baseline_only",
    # "baseline_depth",
    "chl_depth",
    # "surfacekd_depth",
    "chl_zenith_depth",
    "surfacekd_zenith_depth",
    "chl_surfacekd_depth",
    "chl_surfacekd_zenith_depth",
    # "chl_baseline_depth",
    "surfacekd_baseline_depth",
    "surfacekd_zenith_baseline_depth",
    "chl_surfacekd_zenith_baseline_depth",
]

# 是否使用完整叶绿素幂指数网格。
# True 更充分但更慢；False 更快。
USE_FULL_CHL_POWER_GRID = True

include_raw_chl = True
include_log1p_chl = False

if USE_FULL_CHL_POWER_GRID:
    chl_power_candidates = np.round(np.arange(0.6, 2.5, 0.1), 2).tolist()
else:
    chl_power_candidates = [ 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.4]

surface_kd_transform_candidates = [
    "raw",
]

zenith_transform_candidates = [
    "cos",
]

# 为了和后续重建脚本兼容，默认建议 raw。
# 如果启用 norm，需要确保重建脚本也支持相同 depth_transform。
depth_transform_candidates = [
    "raw",
]

n_splines_candidates = [4, 6, 8, 10]
lam_candidates = [30.0, 100.0, 300.0]

use_interaction = True

# 推荐 leave_one_profile_out。
# leave_one_point_out 会使同一剖面的其它深度点进入训练集，不适合作为最终论文指标。
cv_mode = "leave_one_profile_out"

# 推荐优先级：
# 1. "profile_rmse_mean"：每条 profile 的逐点 RMSE，再对 profile 求平均，最适合剖面重建任务
# 2. "point_kd_rmse"：所有深度点合并后的全局 RMSE
# 3. "profile_mean_kd_rmse"：只比较每条剖面的平均 KD，不能反映剖面形状
select_best_by = "profile_rmse_mean"

# 多指标排序时的辅助排序列
secondary_sort_cols = [
    "profile_mae_mean",
    "point_kd_rmse",
    "point_kd_mae",
]

remove_outlier_points = False
outlier_z_threshold = 4.0

make_plot = True
show_plot = False

save_all_grid_predictions = False
save_depthwise_prediction_csv = True
save_profile_metric_csv = True
save_cv_profile_plots = True
max_cv_profile_plots = None     # None 表示保存全部 profile；也可设为 20

PRINT_BEST_MODEL_INFO = True
PRINT_TOP_K_MODELS = 8
SAVE_BEST_MODEL_INFO_CSV = True

CHECK_RESIDUAL_CONSISTENCY = True
RESIDUAL_CONSISTENCY_TOL = 1e-5


# =========================================================
# 4. 叶绿素变换候选
# =========================================================

chl_transform_candidates = []

if include_raw_chl:
    chl_transform_candidates.append({
        "name": "raw",
        "kind": "raw",
        "power": np.nan,
    })

if include_log1p_chl:
    chl_transform_candidates.append({
        "name": "log1p",
        "kind": "log1p",
        "power": np.nan,
    })

for p in chl_power_candidates:
    chl_transform_candidates.append({
        "name": f"pow{p:.2f}",
        "kind": "power",
        "power": float(p),
    })


# =========================================================
# 5. 工具函数
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_first(series):
    s = series.dropna()
    return np.nan if len(s) == 0 else s.iloc[0]


def safe_filename(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(":", "_")


def transform_chl(x, transform_spec):
    x = np.asarray(x, dtype=float)
    kind = transform_spec["kind"]

    if kind == "raw":
        return x

    if kind == "log1p":
        if np.any(x < -1):
            raise ValueError("top_chl 中存在小于 -1 的值，不能使用 log1p")
        return np.log1p(x)

    if kind == "power":
        p = float(transform_spec["power"])
        if np.any(x < 0):
            raise ValueError("top_chl 中存在负值，不能使用幂指数变换")
        return np.power(x, p)

    raise ValueError(f"未知叶绿素变换方式: {kind}")


def transform_surface_kd(x, mode="raw"):
    x = np.asarray(x, dtype=float)

    if mode == "raw":
        return x

    if mode == "log1p":
        if np.any(x < -1):
            raise ValueError("surface_kd 中存在小于 -1 的值，不能使用 log1p")
        return np.log1p(x)

    if mode == "sqrt":
        if np.any(x < 0):
            raise ValueError("surface_kd 中存在负值，不能使用 sqrt")
        return np.sqrt(x)

    raise ValueError(f"未知 surface_kd 变换方式: {mode}")


def transform_zenith(zenith_deg, mode="cos"):
    z = np.asarray(zenith_deg, dtype=float)

    if mode == "deg":
        return z

    if mode == "rad":
        return np.deg2rad(z)

    if mode == "cos":
        return np.cos(np.deg2rad(z))

    if mode == "sec":
        c = np.cos(np.deg2rad(z))
        c = np.clip(c, 1e-6, None)
        return 1.0 / c

    raise ValueError(f"未知 zenith_transform: {mode}")


def transform_depth(depth, mode="raw", depth_range=None):
    d = np.asarray(depth, dtype=float)

    if mode == "raw":
        return d

    if mode in ["norm", "minmax"]:
        if depth_range is not None:
            zmin, zmax = depth_range
            zmin = float(zmin)
            zmax = float(zmax)
        else:
            zmin = np.nanmin(d)
            zmax = np.nanmax(d)

        if not np.isfinite(zmin) or not np.isfinite(zmax) or abs(zmax - zmin) < 1e-12:
            return d

        return (d - zmin) / (zmax - zmin)

    raise ValueError(f"未知 depth_transform: {mode}")


def calc_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calc_regression_metrics(y_true, y_pred):
    """
    以预测值与真实值的差为核心的评价函数。
    corr 仅作为辅助输出。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "median_ae": np.nan,
            "q90_ae": np.nan,
            "max_ae": np.nan,
            "r2": np.nan,
            "corr": np.nan,
        }

    err = y_pred - y_true
    ae = np.abs(err)

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "median_ae": float(np.median(ae)),
        "q90_ae": float(np.quantile(ae, 0.90)),
        "max_ae": float(np.max(ae)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "corr": calc_corr(y_true, y_pred),
    }


def format_value_for_print(v, digits=6):
    if pd.isna(v):
        return "NA"

    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"

    return str(v)


def build_gam(n_splines, lam, n_features, use_interaction=False):
    if n_features < 1:
        raise ValueError("n_features 至少为 1")

    terms = s(0, n_splines=n_splines)

    for i in range(1, n_features):
        terms = terms + s(i, n_splines=n_splines)

    if use_interaction and n_features == 2:
        terms = terms + te(
            0,
            1,
            n_splines=[n_splines, n_splines]
        )

    return LinearGAM(terms, lam=lam)


def cv_predict_gam(
    X,
    y,
    groups,
    n_splines,
    lam,
    use_interaction=False,
    mode="leave_one_profile_out",
):
    y = np.asarray(y, dtype=float)
    y_pred = np.full_like(y, np.nan, dtype=float)

    if mode == "leave_one_profile_out":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, y, groups)

    elif mode == "leave_one_point_out":
        splitter = LeaveOneOut()
        split_iter = splitter.split(X, y)

    else:
        raise ValueError("cv_mode 只能是 'leave_one_profile_out' 或 'leave_one_point_out'")

    for train_idx, test_idx in split_iter:
        gam = build_gam(
            n_splines=n_splines,
            lam=lam,
            n_features=X.shape[1],
            use_interaction=use_interaction,
        )

        gam.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = gam.predict(X[test_idx])

    return y_pred


def find_transform_spec_by_name(name, candidates):
    for spec in candidates:
        if spec["name"] == name:
            return spec
    raise ValueError(f"找不到叶绿素变换方式: {name}")


def normalize_depth_ranges(depth_ranges):
    if depth_ranges is None or len(depth_ranges) == 0:
        return [None]

    out = []

    for item in depth_ranges:
        if item is None:
            out.append(None)
            continue

        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"非法深度范围: {item}，应写成 (zmin, zmax)")

        zmin, zmax = item
        zmin = float(zmin)
        zmax = float(zmax)

        if zmin > zmax:
            zmin, zmax = zmax, zmin

        out.append((zmin, zmax))

    return out


def depth_range_to_label(depth_range):
    if depth_range is None:
        return "all_depth"

    zmin, zmax = depth_range

    def fmt(x):
        s = f"{x:g}"
        return s.replace(".", "p")

    return f"depth_{fmt(zmin)}_{fmt(zmax)}m"


def filter_df_by_depth_range(df_in, depth_col_name, depth_range, include_right=True):
    if depth_range is None:
        return df_in.copy()

    zmin, zmax = depth_range

    if include_right:
        mask = (df_in[depth_col_name] >= zmin) & (df_in[depth_col_name] <= zmax)
    else:
        mask = (df_in[depth_col_name] >= zmin) & (df_in[depth_col_name] < zmax)

    return df_in[mask].copy()


def load_zenith_from_mat(mat_path, struct_name="Zenith_CS", field_name="Zenith"):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"找不到 zenith 文件：{mat_path}")

    mat = scipy.io.loadmat(mat_path)

    if struct_name not in mat:
        raise KeyError(f"{mat_path} 中找不到结构体 {struct_name}")

    struct_arr = mat[struct_name]

    if struct_arr.ndim != 2:
        raise ValueError(f"{struct_name} 维度异常: {struct_arr.shape}")

    records = struct_arr[0]
    zenith_list = []

    for rec in records:
        if field_name not in rec.dtype.names:
            raise KeyError(f"{struct_name} 中找不到字段 {field_name}")

        value = rec[field_name]
        value = np.asarray(value, dtype=float).squeeze()

        if value.size == 0:
            zenith_list.append(np.nan)
        else:
            zenith_list.append(float(value))

    return np.asarray(zenith_list, dtype=float)


def build_zenith_mapping(profile_ids_sorted, zenith_deg_arr, profile_id_order_in_mat=None):
    profile_ids_sorted = list(profile_ids_sorted)
    zenith_deg_arr = np.asarray(zenith_deg_arr, dtype=float)

    if profile_id_order_in_mat is None:
        if len(profile_ids_sorted) != len(zenith_deg_arr):
            raise ValueError(
                f"sorted(profile_id) 数量 = {len(profile_ids_sorted)}，"
                f"但 zenith 数量 = {len(zenith_deg_arr)}，无法默认一一对应。"
            )

        order = profile_ids_sorted

    else:
        order = list(profile_id_order_in_mat)

        if len(order) != len(zenith_deg_arr):
            raise ValueError(
                f"profile_id_order_in_mat 数量 = {len(order)}，"
                f"但 zenith 数量 = {len(zenith_deg_arr)}，不一致。"
            )

    return dict(zip(order, zenith_deg_arr))


# =========================================================
# 6. 特征矩阵
# =========================================================

MODEL_FEATURES = {
    "depth_only": ["depth"],
    "baseline_only": ["baseline"],
    "baseline_depth": ["baseline", "depth"],
    "chl_depth": ["chl", "depth"],
    "surfacekd_depth": ["surfacekd", "depth"],
    "chl_zenith_depth": ["chl", "zenith", "depth"],
    "surfacekd_zenith_depth": ["surfacekd", "zenith", "depth"],
    "chl_surfacekd_depth": ["chl", "surfacekd", "depth"],
    "chl_surfacekd_zenith_depth": ["chl", "surfacekd", "zenith", "depth"],
    "chl_baseline_depth": ["chl", "baseline", "depth"],
    "surfacekd_baseline_depth": ["surfacekd", "baseline", "depth"],
    "surfacekd_zenith_baseline_depth": ["surfacekd", "zenith", "baseline", "depth"],
    "chl_surfacekd_zenith_baseline_depth": ["chl", "surfacekd", "zenith", "baseline", "depth"],
}


def build_feature_matrix(
    df_feature,
    model_mode,
    chl_transform_spec,
    surface_kd_transform_mode,
    zenith_transform_mode,
    depth_transform_mode,
    depth_range=None,
):
    if model_mode not in MODEL_FEATURES:
        raise ValueError(f"未知 model_mode: {model_mode}")

    chl_used = transform_chl(
        df_feature[chl_col].to_numpy(dtype=float),
        chl_transform_spec
    )

    surface_kd_used = transform_surface_kd(
        df_feature["surface_kd"].to_numpy(dtype=float),
        surface_kd_transform_mode
    )

    zenith_used = transform_zenith(
        df_feature["zenith_deg"].to_numpy(dtype=float),
        zenith_transform_mode
    )

    depth_used = transform_depth(
        df_feature[depth_col].to_numpy(dtype=float),
        depth_transform_mode,
        depth_range=depth_range,
    )

    baseline_used = df_feature[baseline_col].to_numpy(dtype=float)

    feature_pool = {
        "chl": chl_used,
        "surfacekd": surface_kd_used,
        "zenith": zenith_used,
        "depth": depth_used,
        "baseline": baseline_used,
    }

    feature_names = MODEL_FEATURES[model_mode]
    X = np.column_stack([feature_pool[name] for name in feature_names])

    transformed_cache = {
        "chl_used": chl_used,
        "surface_kd_used": surface_kd_used,
        "zenith_used": zenith_used,
        "depth_used": depth_used,
        "baseline_used": baseline_used,
    }

    return X, feature_names, transformed_cache


# =========================================================
# 7. 预测结果与误差指标
# =========================================================

def add_prediction_columns(df_pred, target_pred_col):
    out = df_pred.copy()

    if training_target_mode == "depthwise_kd":
        out["true_kd"] = out[kd_col].astype(float)
        out["pred_kd"] = out[target_pred_col].astype(float)

        out["true_residual"] = out[residual_col].astype(float)
        out["pred_residual"] = out["pred_kd"] - out[baseline_col].astype(float)

    elif training_target_mode == "depthwise_residual":
        out["true_kd"] = out[kd_col].astype(float)
        out["true_residual"] = out[residual_col].astype(float)

        out["pred_residual"] = out[target_pred_col].astype(float)
        out["pred_kd"] = out[baseline_col].astype(float) + out["pred_residual"]

    else:
        raise ValueError("training_target_mode 只能是 'depthwise_kd' 或 'depthwise_residual'")

    out["kd_error"] = out["pred_kd"] - out["true_kd"]
    out["kd_abs_error"] = out["kd_error"].abs()

    out["residual_error"] = out["pred_residual"] - out["true_residual"]
    out["residual_abs_error"] = out["residual_error"].abs()

    return out


def build_profile_point_error_metrics(df_pred):
    """
    每条 profile 内部先计算逐点 KD RMSE / MAE。
    这是评估剖面重建质量的核心表。
    """
    records = []

    for pid, g in df_pred.groupby(profile_col):
        g = g.sort_values(depth_col).copy()

        kd_m = calc_regression_metrics(g["true_kd"], g["pred_kd"])
        res_m = calc_regression_metrics(g["true_residual"], g["pred_residual"])

        records.append({
            profile_col: pid,
            "n_depth_points": len(g),
            "depth_min": float(g[depth_col].min()),
            "depth_max": float(g[depth_col].max()),
            "top_chl": safe_first(g[chl_col]),
            "surface_kd": safe_first(g["surface_kd"]),
            "zenith_deg": safe_first(g["zenith_deg"]),

            "kd_rmse": kd_m["rmse"],
            "kd_mae": kd_m["mae"],
            "kd_bias": kd_m["bias"],
            "kd_median_ae": kd_m["median_ae"],
            "kd_q90_ae": kd_m["q90_ae"],
            "kd_corr": kd_m["corr"],

            "residual_rmse": res_m["rmse"],
            "residual_mae": res_m["mae"],
            "residual_bias": res_m["bias"],
            "residual_corr": res_m["corr"],
        })

    profile_df = pd.DataFrame(records)

    summary = {
        "profile_rmse_mean": float(profile_df["kd_rmse"].mean()),
        "profile_rmse_median": float(profile_df["kd_rmse"].median()),
        "profile_rmse_max": float(profile_df["kd_rmse"].max()),
        "profile_mae_mean": float(profile_df["kd_mae"].mean()),
        "profile_mae_median": float(profile_df["kd_mae"].median()),
        "profile_bias_mean": float(profile_df["kd_bias"].mean()),
        "profile_abs_bias_mean": float(profile_df["kd_bias"].abs().mean()),

        "profile_residual_rmse_mean": float(profile_df["residual_rmse"].mean()),
        "profile_residual_mae_mean": float(profile_df["residual_mae"].mean()),
    }

    return profile_df, summary


def build_profile_mean_metrics(df_pred):
    """
    剖面平均值指标。
    这个指标只能反映每条 profile 的平均 KD 是否接近，不能反映剖面形状。
    """
    profile_mean_df = (
        df_pred
        .groupby(profile_col, as_index=False)
        .agg(
            true_kd_mean=("true_kd", "mean"),
            pred_kd_mean=("pred_kd", "mean"),
            true_residual_mean=("true_residual", "mean"),
            pred_residual_mean=("pred_residual", "mean"),
            baseline_mean=(baseline_col, "mean"),
            kd_error_mean=("kd_error", "mean"),
            residual_error_mean=("residual_error", "mean"),
            n_depth_points=(depth_col, "size"),
            depth_min=(depth_col, "min"),
            depth_max=(depth_col, "max"),
            top_chl=(chl_col, safe_first),
            surface_kd=("surface_kd", safe_first),
            zenith_deg=("zenith_deg", safe_first),
        )
    )

    kd_mean_m = calc_regression_metrics(
        profile_mean_df["true_kd_mean"],
        profile_mean_df["pred_kd_mean"]
    )

    res_mean_m = calc_regression_metrics(
        profile_mean_df["true_residual_mean"],
        profile_mean_df["pred_residual_mean"]
    )

    summary = {
        "profile_mean_kd_mae": kd_mean_m["mae"],
        "profile_mean_kd_rmse": kd_mean_m["rmse"],
        "profile_mean_kd_bias": kd_mean_m["bias"],
        "profile_mean_kd_corr": kd_mean_m["corr"],
        "profile_mean_kd_r2": kd_mean_m["r2"],

        "profile_mean_residual_mae": res_mean_m["mae"],
        "profile_mean_residual_rmse": res_mean_m["rmse"],
        "profile_mean_residual_bias": res_mean_m["bias"],
        "profile_mean_residual_corr": res_mean_m["corr"],
        "profile_mean_residual_r2": res_mean_m["r2"],
    }

    return profile_mean_df, summary


def collect_all_metrics(df_pred):
    point_kd = calc_regression_metrics(df_pred["true_kd"], df_pred["pred_kd"])
    point_res = calc_regression_metrics(df_pred["true_residual"], df_pred["pred_residual"])

    profile_point_df, profile_point_summary = build_profile_point_error_metrics(df_pred)
    profile_mean_df, profile_mean_summary = build_profile_mean_metrics(df_pred)

    metric = {
        "point_kd_mae": point_kd["mae"],
        "point_kd_rmse": point_kd["rmse"],
        "point_kd_bias": point_kd["bias"],
        "point_kd_median_ae": point_kd["median_ae"],
        "point_kd_q90_ae": point_kd["q90_ae"],
        "point_kd_max_ae": point_kd["max_ae"],
        "point_kd_r2": point_kd["r2"],
        "point_kd_corr": point_kd["corr"],
        "point_kd_n": point_kd["n"],

        "point_residual_mae": point_res["mae"],
        "point_residual_rmse": point_res["rmse"],
        "point_residual_bias": point_res["bias"],
        "point_residual_corr": point_res["corr"],
    }

    metric.update(profile_point_summary)
    metric.update(profile_mean_summary)

    return metric, profile_point_df, profile_mean_df


# =========================================================
# 8. 绘图函数
# =========================================================

def make_depthwise_scatter_plot(df_pred, save_path, title):
    d = df_pred[["true_kd", "pred_kd"]].dropna().copy()

    if len(d) == 0:
        return

    x = d["true_kd"].to_numpy(dtype=float)
    y = d["pred_kd"].to_numpy(dtype=float)

    m = calc_regression_metrics(x, y)

    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = (hi - lo) * 0.05 if hi > lo else 0.01
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, s=16, alpha=0.45)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Observed depthwise KD")
    ax.set_ylabel("Predicted depthwise KD")
    ax.set_title(
        f"{title}\n"
        f"RMSE={m['rmse']:.5f}, MAE={m['mae']:.5f}, Bias={m['bias']:.5f}, "
        f"R={m['corr']:.3f}, N={m['n']}"
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def make_error_histogram(df_pred, save_path, title):
    d = df_pred["kd_error"].dropna().to_numpy(dtype=float)

    if len(d) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(d, bins=30, alpha=0.8)
    ax.axvline(0, linestyle="--", linewidth=2)

    ax.set_xlabel("Prediction error: pred KD - observed KD")
    ax.set_ylabel("Count")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def make_error_by_depth_plot(df_pred, save_path, title, n_bins=8):
    d = df_pred[[depth_col, "kd_error", "kd_abs_error"]].dropna().copy()

    if len(d) == 0:
        return

    zmin = d[depth_col].min()
    zmax = d[depth_col].max()

    if zmax <= zmin:
        return

    bins = np.linspace(zmin, zmax, n_bins + 1)
    d["depth_bin"] = pd.cut(d[depth_col], bins=bins, include_lowest=True)

    records = []

    for b, g in d.groupby("depth_bin"):
        if len(g) == 0:
            continue

        err = g["kd_error"].to_numpy(dtype=float)

        records.append({
            "depth_mid": float(g[depth_col].mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "n": int(len(g)),
        })

    stat = pd.DataFrame(records)

    if stat.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(stat["depth_mid"], stat["rmse"], marker="o", label="RMSE")
    ax.plot(stat["depth_mid"], stat["mae"], marker="s", label="MAE")
    ax.plot(stat["depth_mid"], stat["bias"], marker="^", label="Bias")

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Error metric")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def make_profile_rmse_bar_plot(profile_error_df, save_path, title):
    if profile_error_df.empty:
        return

    d = profile_error_df.sort_values("kd_rmse", ascending=False).copy()
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(max(8, len(d) * 0.35), 5))
    ax.bar(x, d["kd_rmse"].to_numpy(dtype=float))

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in d[profile_col]], rotation=90)

    ax.set_xlabel("Profile ID")
    ax.set_ylabel("Profile-wise KD RMSE")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def make_profile_mean_scatter_plot(profile_mean_df, save_path, title):
    d = profile_mean_df[["true_kd_mean", "pred_kd_mean", profile_col]].dropna().copy()

    if len(d) == 0:
        return

    x = d["true_kd_mean"].to_numpy(dtype=float)
    y = d["pred_kd_mean"].to_numpy(dtype=float)
    pid = d[profile_col].to_numpy()

    m = calc_regression_metrics(x, y)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = (hi - lo) * 0.05 if hi > lo else 0.01
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, s=50, alpha=0.85)

    for xi, yi, pi in zip(x, y, pid):
        ax.text(xi, yi, str(int(pi)), fontsize=8, ha="left", va="bottom")

    ax.plot([lo, hi], [lo, hi], "--", linewidth=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed profile-mean KD")
    ax.set_ylabel("Predicted profile-mean KD")
    ax.set_title(
        f"{title}\n"
        f"RMSE={m['rmse']:.5f}, MAE={m['mae']:.5f}, Bias={m['bias']:.5f}, R={m['corr']:.3f}"
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def make_cv_profile_plots(df_pred, profile_error_df, save_dir, title_prefix, max_profiles=None):
    ensure_dir(save_dir)

    profile_error_map = {
        row[profile_col]: row
        for _, row in profile_error_df.iterrows()
    }

    profile_ids = sorted(df_pred[profile_col].dropna().unique().tolist())

    if max_profiles is not None:
        profile_ids = profile_ids[:max_profiles]

    for pid in profile_ids:
        g = df_pred[df_pred[profile_col] == pid].sort_values(depth_col).copy()

        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 8))

        ax.plot(
            g["true_kd"],
            g[depth_col],
            linewidth=2,
            label="Observed KD"
        )

        ax.plot(
            g["pred_kd"],
            g[depth_col],
            "--",
            linewidth=2,
            label="Predicted KD"
        )

        ax.invert_yaxis()
        ax.set_xlabel("KD")
        ax.set_ylabel("Depth")

        row = profile_error_map.get(pid, None)

        if row is not None:
            ax.set_title(
                f"{title_prefix} | Profile {pid}\n"
                f"RMSE={row['kd_rmse']:.5f}, MAE={row['kd_mae']:.5f}, Bias={row['kd_bias']:.5f}"
            )
        else:
            ax.set_title(f"{title_prefix} | Profile {pid}")

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        fig_path = os.path.join(
            save_dir,
            f"profile_{safe_filename(pid)}_cv_reconstruction.png"
        )

        plt.savefig(fig_path, dpi=220, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


# =========================================================
# 9. 打印函数
# =========================================================

def print_best_grid_info(eval_df, best, range_label, sort_cols, top_k=8):
    print("\n" + "=" * 100)
    print("当前深度段最优组合")
    print("=" * 100)

    print(f"target_branch        : {target_branch}")
    print(f"training_target_mode : {training_target_mode}")
    print(f"depth_range_label    : {range_label}")
    print(f"cv_mode              : {cv_mode}")
    print(f"select_best_by       : {select_best_by}")
    print(f"best {select_best_by:<22}: {format_value_for_print(best[select_best_by])}")

    print("\n[最优超参数组合]")
    show_cols = [
        "model_mode",
        "feature_names",
        "chl_transform",
        "chl_transform_kind",
        "chl_power",
        "surface_kd_transform",
        "zenith_transform",
        "depth_transform",
        "n_splines",
        "lam",
    ]

    for col in show_cols:
        if col in best.index:
            print(f"{col:<32}: {format_value_for_print(best[col])}")

    print("\n[误差优先指标]")
    metric_cols = [
        "profile_rmse_mean",
        "profile_rmse_median",
        "profile_rmse_max",
        "profile_mae_mean",
        "profile_mae_median",
        "profile_bias_mean",
        "profile_abs_bias_mean",
        "point_kd_rmse",
        "point_kd_mae",
        "point_kd_bias",
        "point_kd_median_ae",
        "point_kd_q90_ae",
        "point_kd_max_ae",
        "profile_mean_kd_rmse",
        "profile_mean_kd_mae",
        "point_kd_corr",
        "point_kd_r2",
        "n_profiles",
        "n_depth_points",
    ]

    for col in metric_cols:
        if col in best.index:
            print(f"{col:<32}: {format_value_for_print(best[col])}")

    print(f"\n[前 {top_k} 个候选组合]")
    top_cols = [
        "model_mode",
        "chl_transform",
        "surface_kd_transform",
        "zenith_transform",
        "depth_transform",
        "n_splines",
        "lam",
    ] + sort_cols + [
        "point_kd_corr",
        "profile_mean_kd_rmse",
    ]

    top_cols = [c for c in top_cols if c in eval_df.columns]
    top_df = eval_df[top_cols].head(top_k).copy()

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 80,
    ):
        print(top_df.to_string(index=False))

    print("=" * 100 + "\n")


def print_cv_metric_summary(range_label, metric):
    print("\n" + "-" * 100)
    print("最优组合 CV 验证指标：误差优先")
    print("-" * 100)
    print(f"depth_range_label: {range_label}")

    print("\n[核心指标：每条剖面的逐点误差，然后对 profile 求平均]")
    print(f"profile_rmse_mean   : {format_value_for_print(metric['profile_rmse_mean'])}")
    print(f"profile_rmse_median : {format_value_for_print(metric['profile_rmse_median'])}")
    print(f"profile_mae_mean    : {format_value_for_print(metric['profile_mae_mean'])}")
    print(f"profile_bias_mean   : {format_value_for_print(metric['profile_bias_mean'])}")

    print("\n[全局逐深度点误差]")
    print(f"point_kd_RMSE       : {format_value_for_print(metric['point_kd_rmse'])}")
    print(f"point_kd_MAE        : {format_value_for_print(metric['point_kd_mae'])}")
    print(f"point_kd_Bias       : {format_value_for_print(metric['point_kd_bias'])}")
    print(f"point_kd_MedianAE   : {format_value_for_print(metric['point_kd_median_ae'])}")
    print(f"point_kd_Q90AE      : {format_value_for_print(metric['point_kd_q90_ae'])}")
    print(f"point_kd_R          : {format_value_for_print(metric['point_kd_corr'])}")
    print(f"point_kd_R2         : {format_value_for_print(metric['point_kd_r2'])}")

    print("\n[剖面平均值误差，仅作为后处理辅助指标]")
    print(f"profile_mean_kd_RMSE: {format_value_for_print(metric['profile_mean_kd_rmse'])}")
    print(f"profile_mean_kd_MAE : {format_value_for_print(metric['profile_mean_kd_mae'])}")
    print(f"profile_mean_kd_Bias: {format_value_for_print(metric['profile_mean_kd_bias'])}")
    print(f"profile_mean_kd_R   : {format_value_for_print(metric['profile_mean_kd_corr'])}")
    print("-" * 100 + "\n")


# =========================================================
# 10. 读取数据
# =========================================================

df = pd.read_csv(input_csv)

required_cols = [
    profile_col,
    chl_col,
    depth_col,
    kd_col,
    y_col,
    baseline_col,
    residual_col,
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"输入文件缺少必要列: {missing_cols}")

df = df.dropna(
    subset=[
        profile_col,
        chl_col,
        depth_col,
        kd_col,
        y_col,
        baseline_col,
        residual_col,
    ]
).copy()

if exclude_profile_ids:
    df = df[~df[profile_col].isin(exclude_profile_ids)].copy()

df[profile_col] = df[profile_col].astype(int)
df[depth_col] = df[depth_col].astype(float)

print("读取数据完成")
print("target_branch =", target_branch)
print("training_target_mode =", training_target_mode)
print("使用的 kd_col =", kd_col)
print("使用的 baseline_col =", baseline_col)
print("使用的 residual_col =", residual_col)
print("逐深度数据维度：", df.shape)
print("profile 数量：", df[profile_col].nunique())


# =========================================================
# 11. 提取 surface_kd
# =========================================================

surface_kd_df = (
    df.sort_values([profile_col, depth_col])
    .groupby(profile_col, as_index=False)
    .first()[[profile_col, depth_col, kd_col]]
    .rename(columns={
        depth_col: "surface_depth",
        kd_col: "surface_kd",
    })
)

print("\n提取的 surface_kd：")
print(surface_kd_df.head())


# =========================================================
# 12. 读取 zenith.mat
# =========================================================

profile_ids_sorted = sorted(df[profile_col].unique().tolist())

zenith_deg_arr = load_zenith_from_mat(
    zenith_mat_path,
    struct_name="Zenith_CS",
    field_name="Zenith"
)

zenith_mapping = build_zenith_mapping(
    profile_ids_sorted,
    zenith_deg_arr,
    profile_id_order_in_mat
)

print("\n已读取 zenith.mat")
print("天顶角数量：", len(zenith_deg_arr))


# =========================================================
# 13. 单深度段训练函数
# =========================================================

def run_one_depth_range_analysis_depthwise(df_all, depth_range, include_right):
    range_label = depth_range_to_label(depth_range)
    range_output_dir = os.path.join(output_dir, range_label)
    ensure_dir(range_output_dir)

    plot_dir = os.path.join(range_output_dir, "plots")
    ensure_dir(plot_dir)

    profile_plot_dir = os.path.join(range_output_dir, "cv_profile_plots")
    ensure_dir(profile_plot_dir)

    eval_csv = os.path.join(
        range_output_dir,
        f"gam_depthwise_gridsearch_eval_{training_target_mode}_{target_branch}.csv"
    )

    all_pred_csv = os.path.join(
        range_output_dir,
        f"all_gam_depthwise_predictions_{training_target_mode}_{target_branch}.csv"
    )

    best_cv_pred_csv = os.path.join(
        range_output_dir,
        f"best_depthwise_cv_prediction_{training_target_mode}_{target_branch}.csv"
    )

    finalfit_pred_csv = os.path.join(
        range_output_dir,
        f"finalfit_depthwise_prediction_{training_target_mode}_{target_branch}.csv"
    )

    profile_point_cv_csv = os.path.join(
        range_output_dir,
        f"profile_point_error_cv_{training_target_mode}_{target_branch}.csv"
    )

    profile_mean_cv_csv = os.path.join(
        range_output_dir,
        f"profile_mean_metric_cv_{training_target_mode}_{target_branch}.csv"
    )

    profile_point_finalfit_csv = os.path.join(
        range_output_dir,
        f"profile_point_error_finalfit_{training_target_mode}_{target_branch}.csv"
    )

    profile_mean_finalfit_csv = os.path.join(
        range_output_dir,
        f"profile_mean_metric_finalfit_{training_target_mode}_{target_branch}.csv"
    )

    best_model_path = os.path.join(
        range_output_dir,
        f"best_depthwise_model_{training_target_mode}_{target_branch}.joblib"
    )

    df_range = filter_df_by_depth_range(
        df_all,
        depth_col,
        depth_range,
        include_right=include_right,
    )

    print("\n" + "=" * 80)
    print(f"[{training_target_mode}] 当前分析范围：{depth_range}, include_right={include_right}")
    print("该范围逐深度数据维度：", df_range.shape)

    if df_range.empty:
        print("该深度范围内没有数据，跳过。")
        return None

    train_df = df_range.copy()

    train_df = train_df.merge(
        surface_kd_df,
        on=profile_col,
        how="left"
    )

    train_df["zenith_deg"] = train_df[profile_col].map(zenith_mapping)

    train_df = train_df.dropna(
        subset=[
            chl_col,
            "surface_kd",
            "zenith_deg",
            depth_col,
            kd_col,
            baseline_col,
            residual_col,
        ]
    ).copy()

    count_df = (
        train_df
        .groupby(profile_col, as_index=False)
        .agg(n_depth_points=(depth_col, "size"))
    )

    good_profiles = count_df[
        count_df["n_depth_points"] >= min_points_per_profile_in_range
    ][profile_col].tolist()

    train_df = train_df[
        train_df[profile_col].isin(good_profiles)
    ].copy()

    if train_df[profile_col].nunique() < 3:
        print("该范围内可用于建模的 profile 太少，跳过。")
        return None

    if training_target_mode == "depthwise_kd":
        train_df["target_true"] = train_df[kd_col].astype(float)

    elif training_target_mode == "depthwise_residual":
        train_df["target_true"] = train_df[residual_col].astype(float)

        if CHECK_RESIDUAL_CONSISTENCY:
            check_err = (
                train_df[kd_col].astype(float)
                - train_df[baseline_col].astype(float)
                - train_df[residual_col].astype(float)
            )

            max_abs_check_err = float(np.nanmax(np.abs(check_err)))

            if max_abs_check_err > RESIDUAL_CONSISTENCY_TOL:
                print(
                    "\n警告：residual_col 与 kd_col - baseline_col 不完全一致。"
                    f" max_abs(kd - baseline - residual) = {max_abs_check_err:.6g}"
                )
                print("如果 baseline/residual 与 kd 不在同一量纲，pred_kd = baseline + pred_residual 的解释需要重新确认。\n")

    else:
        raise ValueError("training_target_mode 只能是 'depthwise_kd' 或 'depthwise_residual'")

    if remove_outlier_points:
        std_val = train_df["target_true"].std()

        if pd.notna(std_val) and std_val > 0:
            z_score = (
                train_df["target_true"]
                - train_df["target_true"].mean()
            ) / std_val

            before_n = len(train_df)
            train_df = train_df[
                np.abs(z_score) <= outlier_z_threshold
            ].copy()

            print(f"异常点过滤：{before_n} -> {len(train_df)}")

    y = train_df["target_true"].to_numpy(dtype=float)
    groups = train_df[profile_col].to_numpy()

    print("用于训练的逐深度点数：", len(train_df))
    print("用于训练的 profile 数：", train_df[profile_col].nunique())

    if cv_mode == "leave_one_point_out":
        print("\n警告：当前使用 leave_one_point_out，会导致同一 profile 的其它深度点进入训练集，不建议作为最终结果。\n")

    eval_records = []
    pred_records = []

    total_count = (
        len(model_mode_candidates)
        * len(chl_transform_candidates)
        * len(surface_kd_transform_candidates)
        * len(zenith_transform_candidates)
        * len(depth_transform_candidates)
        * len(n_splines_candidates)
        * len(lam_candidates)
    )

    now_count = 0

    for model_mode in model_mode_candidates:
        for chl_transform_spec in chl_transform_candidates:
            for surface_kd_transform_mode in surface_kd_transform_candidates:
                for zenith_transform_mode in zenith_transform_candidates:
                    for depth_transform_mode in depth_transform_candidates:

                        try:
                            X, feature_names, transformed_cache = build_feature_matrix(
                                train_df,
                                model_mode,
                                chl_transform_spec,
                                surface_kd_transform_mode,
                                zenith_transform_mode,
                                depth_transform_mode,
                                depth_range=depth_range,
                            )
                        except Exception as e:
                            print(f"跳过特征组合：model_mode={model_mode}, 原因：{e}")
                            continue

                        if np.isnan(X).any() or np.isinf(X).any():
                            print(f"跳过：X 中存在 NaN 或 Inf，model_mode={model_mode}")
                            continue

                        for n_splines in n_splines_candidates:
                            for lam in lam_candidates:
                                now_count += 1

                                try:
                                    y_pred_cv = cv_predict_gam(
                                        X,
                                        y,
                                        groups,
                                        n_splines,
                                        lam,
                                        use_interaction=use_interaction,
                                        mode=cv_mode,
                                    )
                                except Exception as e:
                                    print(
                                        f"[{now_count}/{total_count}] 跳过："
                                        f"mode={model_mode}, s={n_splines}, lam={lam}, 原因：{e}"
                                    )
                                    continue

                                tmp_eval = train_df.copy()
                                tmp_eval["target_pred_cv"] = y_pred_cv
                                tmp_eval["target_error_cv"] = tmp_eval["target_pred_cv"] - tmp_eval["target_true"]

                                tmp_eval = add_prediction_columns(
                                    tmp_eval,
                                    target_pred_col="target_pred_cv"
                                )

                                metric, profile_point_df, profile_mean_df = collect_all_metrics(tmp_eval)

                                record = {
                                    "target_branch": target_branch,
                                    "training_target_mode": training_target_mode,
                                    "depth_range_label": range_label,
                                    "depth_range_min": np.nan if depth_range is None else depth_range[0],
                                    "depth_range_max": np.nan if depth_range is None else depth_range[1],
                                    "include_right": include_right,
                                    "cv_mode": cv_mode,

                                    "model_mode": model_mode,
                                    "feature_names": ",".join(feature_names),
                                    "chl_transform": chl_transform_spec["name"],
                                    "chl_transform_kind": chl_transform_spec["kind"],
                                    "chl_power": chl_transform_spec["power"],
                                    "surface_kd_transform": surface_kd_transform_mode,
                                    "zenith_transform": zenith_transform_mode,
                                    "depth_transform": depth_transform_mode,
                                    "n_splines": n_splines,
                                    "lam": lam,

                                    "n_depth_points": len(train_df),
                                    "n_profiles": train_df[profile_col].nunique(),
                                }

                                record.update(metric)
                                eval_records.append(record)

                                if save_all_grid_predictions:
                                    tmp_pred = train_df.copy()
                                    tmp_pred["model_mode"] = model_mode
                                    tmp_pred["chl_transform"] = chl_transform_spec["name"]
                                    tmp_pred["chl_transform_kind"] = chl_transform_spec["kind"]
                                    tmp_pred["chl_power"] = chl_transform_spec["power"]
                                    tmp_pred["surface_kd_transform"] = surface_kd_transform_mode
                                    tmp_pred["zenith_transform"] = zenith_transform_mode
                                    tmp_pred["depth_transform"] = depth_transform_mode
                                    tmp_pred["n_splines"] = n_splines
                                    tmp_pred["lam"] = lam
                                    tmp_pred["target_pred_cv"] = y_pred_cv
                                    tmp_pred["target_error_cv"] = tmp_pred["target_pred_cv"] - tmp_pred["target_true"]
                                    pred_records.append(tmp_pred)

                                print(
                                    f"[{now_count}/{total_count}] {training_target_mode} | {target_branch} | {range_label} | "
                                    f"mode={model_mode} | chl={chl_transform_spec['name']} | "
                                    f"skd={surface_kd_transform_mode} | zen={zenith_transform_mode} | "
                                    f"depth={depth_transform_mode} | s={n_splines} | lam={lam} | "
                                    f"profile_RMSE_mean={metric['profile_rmse_mean']:.6f}, "
                                    f"point_RMSE={metric['point_kd_rmse']:.6f}, "
                                    f"point_MAE={metric['point_kd_mae']:.6f}, "
                                    f"R={metric['point_kd_corr']:.3f}"
                                )

    eval_df = pd.DataFrame(eval_records)

    if eval_df.empty:
        print("没有成功训练任何模型。")
        return None

    if select_best_by not in eval_df.columns:
        raise ValueError(f"select_best_by={select_best_by} 不在 eval_df 中。")

    sort_cols = [select_best_by] + [c for c in secondary_sort_cols if c in eval_df.columns and c != select_best_by]

    eval_df = (
        eval_df
        .sort_values(sort_cols, ascending=[True] * len(sort_cols))
        .reset_index(drop=True)
    )

    eval_df.to_csv(
        eval_csv,
        index=False,
        encoding="utf-8-sig"
    )

    if save_all_grid_predictions and len(pred_records) > 0:
        all_pred_df = pd.concat(pred_records, ignore_index=True)
        all_pred_df.to_csv(
            all_pred_csv,
            index=False,
            encoding="utf-8-sig"
        )

    best = eval_df.iloc[0]

    if PRINT_BEST_MODEL_INFO:
        print_best_grid_info(
            eval_df=eval_df,
            best=best,
            range_label=range_label,
            sort_cols=sort_cols,
            top_k=PRINT_TOP_K_MODELS,
        )

    if SAVE_BEST_MODEL_INFO_CSV:
        best_info_csv = os.path.join(
            range_output_dir,
            f"best_model_info_{training_target_mode}_{target_branch}.csv"
        )

        topk_info_csv = os.path.join(
            range_output_dir,
            f"top{PRINT_TOP_K_MODELS}_model_info_{training_target_mode}_{target_branch}.csv"
        )

        best.to_frame().T.to_csv(
            best_info_csv,
            index=False,
            encoding="utf-8-sig"
        )

        eval_df.head(PRINT_TOP_K_MODELS).to_csv(
            topk_info_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print("最优组合信息已保存：", best_info_csv)
        print(f"前 {PRINT_TOP_K_MODELS} 个候选组合已保存：", topk_info_csv)

    best_model_mode = best["model_mode"]
    best_chl_transform = best["chl_transform"]
    best_surface_kd_transform = best["surface_kd_transform"]
    best_zenith_transform = best["zenith_transform"]
    best_depth_transform = best["depth_transform"]
    best_n_splines = int(best["n_splines"])
    best_lam = float(best["lam"])

    best_chl_transform_spec = find_transform_spec_by_name(
        best_chl_transform,
        chl_transform_candidates
    )

    X_best, best_feature_names, _ = build_feature_matrix(
        train_df,
        best_model_mode,
        best_chl_transform_spec,
        best_surface_kd_transform,
        best_zenith_transform,
        best_depth_transform,
        depth_range=depth_range,
    )

    # -----------------------------------------------------
    # 最佳组合 CV 预测
    # -----------------------------------------------------
    best_cv_pred = cv_predict_gam(
        X_best,
        y,
        groups,
        best_n_splines,
        best_lam,
        use_interaction=use_interaction,
        mode=cv_mode,
    )

    best_cv_df = train_df.copy()
    best_cv_df["target_pred_cv"] = best_cv_pred
    best_cv_df["target_error_cv"] = best_cv_df["target_pred_cv"] - best_cv_df["target_true"]

    best_cv_df = add_prediction_columns(
        best_cv_df,
        target_pred_col="target_pred_cv"
    )

    cv_metric, profile_point_cv_df, profile_mean_cv_df = collect_all_metrics(best_cv_df)

    if save_depthwise_prediction_csv:
        best_cv_df.to_csv(
            best_cv_pred_csv,
            index=False,
            encoding="utf-8-sig"
        )

    if save_profile_metric_csv:
        profile_point_cv_df.to_csv(
            profile_point_cv_csv,
            index=False,
            encoding="utf-8-sig"
        )

        profile_mean_cv_df.to_csv(
            profile_mean_cv_csv,
            index=False,
            encoding="utf-8-sig"
        )

    # -----------------------------------------------------
    # 用全部逐深度点训练最终模型，用于后续实际重建
    # -----------------------------------------------------
    final_gam = build_gam(
        best_n_splines,
        best_lam,
        X_best.shape[1],
        use_interaction=use_interaction
    )

    final_gam.fit(X_best, y)

    target_label = (
        kd_col
        if training_target_mode == "depthwise_kd"
        else residual_col
    )

    joblib.dump(
        {
            "model": final_gam,
            "training_target_mode": training_target_mode,
            "target_branch": target_branch,
            "target_kd_col": kd_col,
            "target_y_col": y_col,
            "target_residual_col": residual_col,
            "baseline_col": baseline_col,

            "depth_range_label": range_label,
            "depth_range_min": None if depth_range is None else depth_range[0],
            "depth_range_max": None if depth_range is None else depth_range[1],
            "include_right": include_right,

            "model_mode": best_model_mode,
            "feature_names": list(best_feature_names),
            "chl_transform": best_chl_transform,
            "chl_transform_kind": best_chl_transform_spec["kind"],
            "chl_power": best_chl_transform_spec["power"],
            "surface_kd_transform": best_surface_kd_transform,
            "zenith_transform": best_zenith_transform,
            "depth_transform": best_depth_transform,

            "n_splines": best_n_splines,
            "lam": best_lam,
            "input": best_model_mode,
            "target": target_label,

            "cv_mode": cv_mode,
            "select_best_by": select_best_by,
            "secondary_sort_cols": secondary_sort_cols,

            "cv_metrics": cv_metric,
        },
        best_model_path
    )

    print("最佳模型已保存：", best_model_path)

    # -----------------------------------------------------
    # final-fit 预测：训练集拟合结果，不作为泛化能力评价
    # -----------------------------------------------------
    finalfit_df = train_df.copy()
    finalfit_df["target_pred_finalfit"] = final_gam.predict(X_best)
    finalfit_df["target_error_finalfit"] = (
        finalfit_df["target_pred_finalfit"]
        - finalfit_df["target_true"]
    )

    finalfit_df = add_prediction_columns(
        finalfit_df,
        target_pred_col="target_pred_finalfit"
    )

    finalfit_metric, profile_point_finalfit_df, profile_mean_finalfit_df = collect_all_metrics(finalfit_df)

    if save_depthwise_prediction_csv:
        finalfit_df.to_csv(
            finalfit_pred_csv,
            index=False,
            encoding="utf-8-sig"
        )

    if save_profile_metric_csv:
        profile_point_finalfit_df.to_csv(
            profile_point_finalfit_csv,
            index=False,
            encoding="utf-8-sig"
        )

        profile_mean_finalfit_df.to_csv(
            profile_mean_finalfit_csv,
            index=False,
            encoding="utf-8-sig"
        )

    # -----------------------------------------------------
    # 绘图
    # -----------------------------------------------------
    if make_plot:
        make_depthwise_scatter_plot(
            best_cv_df,
            os.path.join(
                plot_dir,
                f"scatter_depthwise_kd_cv_{training_target_mode}_{target_branch}.png"
            ),
            title=f"CV depthwise KD | {target_branch} | {range_label}"
        )

        make_error_histogram(
            best_cv_df,
            os.path.join(
                plot_dir,
                f"hist_kd_error_cv_{training_target_mode}_{target_branch}.png"
            ),
            title=f"CV KD error distribution | {target_branch} | {range_label}"
        )

        make_error_by_depth_plot(
            best_cv_df,
            os.path.join(
                plot_dir,
                f"error_by_depth_cv_{training_target_mode}_{target_branch}.png"
            ),
            title=f"CV error by depth | {target_branch} | {range_label}"
        )

        make_profile_rmse_bar_plot(
            profile_point_cv_df,
            os.path.join(
                plot_dir,
                f"profile_rmse_bar_cv_{training_target_mode}_{target_branch}.png"
            ),
            title=f"CV profile-wise RMSE | {target_branch} | {range_label}"
        )

        make_profile_mean_scatter_plot(
            profile_mean_cv_df,
            os.path.join(
                plot_dir,
                f"scatter_profile_mean_kd_cv_{training_target_mode}_{target_branch}.png"
            ),
            title=f"CV profile-mean KD | {target_branch} | {range_label}"
        )

        if save_cv_profile_plots:
            make_cv_profile_plots(
                best_cv_df,
                profile_point_cv_df,
                profile_plot_dir,
                title_prefix=f"CV {target_branch} {range_label}",
                max_profiles=max_cv_profile_plots,
            )

    if PRINT_BEST_MODEL_INFO:
        print_cv_metric_summary(
            range_label=range_label,
            metric=cv_metric,
        )

    return {
        "training_target_mode": training_target_mode,
        "target_branch": target_branch,
        "depth_range_label": range_label,
        "depth_range_min": np.nan if depth_range is None else depth_range[0],
        "depth_range_max": np.nan if depth_range is None else depth_range[1],
        "include_right": include_right,

        "n_profiles": int(train_df[profile_col].nunique()),
        "n_depth_points": int(len(train_df)),

        "best_model_mode": best_model_mode,
        "best_feature_names": ",".join(best_feature_names),
        "best_chl_transform": best_chl_transform,
        "best_chl_power": float(best["chl_power"]) if pd.notna(best["chl_power"]) else np.nan,
        "best_surface_kd_transform": best_surface_kd_transform,
        "best_zenith_transform": best_zenith_transform,
        "best_depth_transform": best_depth_transform,
        "best_n_splines": best_n_splines,
        "best_lam": best_lam,

        "cv_mode": cv_mode,
        "select_best_by": select_best_by,

        "cv_profile_rmse_mean": cv_metric["profile_rmse_mean"],
        "cv_profile_rmse_median": cv_metric["profile_rmse_median"],
        "cv_profile_mae_mean": cv_metric["profile_mae_mean"],
        "cv_profile_bias_mean": cv_metric["profile_bias_mean"],

        "cv_point_kd_rmse": cv_metric["point_kd_rmse"],
        "cv_point_kd_mae": cv_metric["point_kd_mae"],
        "cv_point_kd_bias": cv_metric["point_kd_bias"],
        "cv_point_kd_median_ae": cv_metric["point_kd_median_ae"],
        "cv_point_kd_q90_ae": cv_metric["point_kd_q90_ae"],
        "cv_point_kd_corr": cv_metric["point_kd_corr"],
        "cv_point_kd_r2": cv_metric["point_kd_r2"],

        "cv_profile_mean_kd_rmse": cv_metric["profile_mean_kd_rmse"],
        "cv_profile_mean_kd_mae": cv_metric["profile_mean_kd_mae"],
        "cv_profile_mean_kd_corr": cv_metric["profile_mean_kd_corr"],

        "eval_csv": eval_csv,
        "best_cv_pred_csv": best_cv_pred_csv,
        "finalfit_pred_csv": finalfit_pred_csv,
        "profile_point_cv_csv": profile_point_cv_csv,
        "profile_mean_cv_csv": profile_mean_cv_csv,
        "best_model_path": best_model_path,
        "output_dir": range_output_dir,
    }


# =========================================================
# 14. 主调度
# =========================================================

depth_ranges = normalize_depth_ranges(depth_ranges)
summary_records = []

for idx, one_range in enumerate(depth_ranges):
    include_right = (
        idx == len(depth_ranges) - 1
        and last_range_include_right
    )

    result = run_one_depth_range_analysis_depthwise(
        df,
        one_range,
        include_right=include_right,
    )

    if result is not None:
        summary_records.append(result)

if len(summary_records) > 0:
    summary_df = pd.DataFrame(summary_records)

    summary_csv = os.path.join(
        output_dir,
        f"depth_range_training_summary_{training_target_mode}_{target_branch}.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 120)
    print("全部深度范围训练汇总：误差优先")
    print("=" * 120)

    summary_show_cols = [
        "training_target_mode",
        "target_branch",
        "depth_range_label",
        "include_right",
        "n_profiles",
        "n_depth_points",

        "best_model_mode",
        "best_feature_names",
        "best_chl_transform",
        "best_chl_power",
        "best_surface_kd_transform",
        "best_zenith_transform",
        "best_depth_transform",
        "best_n_splines",
        "best_lam",

        "cv_mode",
        "select_best_by",

        "cv_profile_rmse_mean",
        "cv_profile_rmse_median",
        "cv_profile_mae_mean",
        "cv_profile_bias_mean",

        "cv_point_kd_rmse",
        "cv_point_kd_mae",
        "cv_point_kd_bias",
        "cv_point_kd_median_ae",
        "cv_point_kd_q90_ae",
        "cv_point_kd_corr",
        "cv_point_kd_r2",

        "cv_profile_mean_kd_rmse",
        "cv_profile_mean_kd_mae",
        "cv_profile_mean_kd_corr",
    ]

    summary_show_cols = [
        c for c in summary_show_cols
        if c in summary_df.columns
    ]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", 80,
    ):
        print(summary_df[summary_show_cols].to_string(index=False))

    print("\n汇总文件已保存：", summary_csv)
    print("=" * 120)

else:
    print("\n没有任何深度范围成功完成分析。")

print("\n全部完成。输出目录：")
print(output_dir)