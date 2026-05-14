import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from pygam import LinearGAM, s, te
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# revised_profile_training_with_baseline_switch.py
#
# 这版把“是否使用深度基线”做成了明确开关，适合 slow / fast 都训练：
#
# training_target_mode 可选：
# 1) "residual"
#    使用基线
#    target = mean_z[ b_i(z) ] = mean_z[ Y_i(z) - B(z) ]
#
# 2) "direct_y"
#    不使用基线
#    target = mean_z[ Y_i(z) ]
#
# 3) "ac"
#    使用基线，但不是直接学残差，而是拟合：
#        Y_i(z) ≈ a_i * B(z) + c_i
#    再训练 a_i 和 c_i
#
# 推荐：
# - Kd_slow: 先试 residual / direct_y
# - Kd_fast: 优先试 direct_y / ac
# =========================================================


# =========================================================
# 1. 路径设置
# =========================================================

input_csv = r"D:\work\gam_b_depthwise_data\interp_kd_chl_density_depthwise_b_for_gam.csv"
zenith_mat_path = r"D:\work\zenith.mat"

# 可选："slow" / "fast"
target_branch = "fast"

# 可选：
# "residual" -> 使用基线，训练平均残差
# "direct_y" -> 不使用基线，训练平均 Y
# "ac"       -> 使用基线，训练 a/c 参数
training_target_mode = "ac"

output_dir = rf"D:\work\gam_profile_training_{training_target_mode}_with_surfacekd_zenith_{target_branch}"
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
    target_depthwise_col = "b_slow_depthwise"
elif target_branch == "fast":
    kd_col = "kd_fast_interp"
    y_col = "Y_fast_interp"
    baseline_col = "B_y_fast"
    target_depthwise_col = "b_fast_depthwise"
else:
    raise ValueError("target_branch 只能是 'slow' 或 'fast'")


# =========================================================
# 3. 可调参数
# =========================================================

exclude_profile_ids = []

depth_ranges = [
    # (0, 15),(10,80),
    (0,80)
]

min_points_per_profile_in_range = 3
profile_id_order_in_mat = None

model_mode_candidates = [
    "chl_only",
    "surfacekd_only",
    # "zenith_only",
    "chl_zenith",
    "surfacekd_zenith",
    "chl_surfacekd",
    "chl_surfacekd_zenith",
]

# 叶绿素变换
include_raw_chl = True
include_log1p_chl = False
chl_power_candidates = np.round(np.arange(0.5, 2.5, 0.10), 2).tolist()

# surface_kd 变换
surface_kd_transform_candidates = [
    "raw",
    # "log1p",
    # "sqrt",
]

# zenith 变换
zenith_transform_candidates = [
    "cos",
]

# GAM 参数
n_splines_candidates = [4, 6, 8, 10, 12]
lam_candidates = [
    # 1.0, 3.0, 10.0,
                  30.0, 100.0, 300.0]

use_interaction = True

remove_outlier_profiles = False
outlier_z_threshold = 3.0
make_plot = True

# 仅 ac 模式：限制 a 的预测范围，避免极端值
clip_a_pred = True
clip_a_min = 0.2
clip_a_max = 3.0

# 是否保存逐深度重建/回填结果
save_depthwise_reconstruction_csv = True


# =========================================================
# 4. 叶绿素变换候选列表
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

def transform_chl(x, transform_spec):
    x = np.asarray(x, dtype=float)
    kind = transform_spec["kind"]

    if kind == "raw":
        return x
    elif kind == "log1p":
        if np.any(x < -1):
            raise ValueError("top_chl 中存在小于 -1 的值，不能使用 log1p")
        return np.log1p(x)
    elif kind == "power":
        p = float(transform_spec["power"])
        if np.any(x < 0):
            raise ValueError("top_chl 中存在负值，不能使用幂指数变换")
        return np.power(x, p)
    else:
        raise ValueError(f"未知叶绿素变换方式: {kind}")


def transform_surface_kd(x, mode="raw"):
    x = np.asarray(x, dtype=float)

    if mode == "raw":
        return x
    elif mode == "log1p":
        if np.any(x < -1):
            raise ValueError("surface_kd 中存在小于 -1 的值，不能使用 log1p")
        return np.log1p(x)
    elif mode == "sqrt":
        if np.any(x < 0):
            raise ValueError("surface_kd 中存在负值，不能使用 sqrt")
        return np.sqrt(x)
    else:
        raise ValueError(f"未知 surface_kd 变换方式: {mode}")


def transform_zenith(zenith_deg, mode="deg"):
    z = np.asarray(zenith_deg, dtype=float)

    if mode == "deg":
        return z
    elif mode == "rad":
        return np.deg2rad(z)
    elif mode == "cos":
        return np.cos(np.deg2rad(z))
    elif mode == "sec":
        c = np.cos(np.deg2rad(z))
        c = np.clip(c, 1e-6, None)
        return 1.0 / c
    else:
        raise ValueError(f"未知 zenith_transform: {mode}")


def calc_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "corr": calc_corr(y_true, y_pred),
    }


def build_gam(n_splines, lam, n_features, use_interaction=False):
    if n_features == 1:
        return LinearGAM(s(0, n_splines=n_splines), lam=lam)

    elif n_features == 2:
        if use_interaction:
            return LinearGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines)
                + te(0, 1, n_splines=[n_splines, n_splines]),
                lam=lam,
            )
        else:
            return LinearGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines),
                lam=lam,
            )

    elif n_features == 3:
        return LinearGAM(
            s(0, n_splines=n_splines)
            + s(1, n_splines=n_splines)
            + s(2, n_splines=n_splines),
            lam=lam,
        )

    else:
        raise ValueError(f"不支持的特征维度: {n_features}")


def loocv_predict_gam(X, y, n_splines, lam, use_interaction=False):
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
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


def filter_df_by_depth_range(df_in, depth_col, depth_range):
    if depth_range is None:
        return df_in.copy()

    zmin, zmax = depth_range
    return df_in[(df_in[depth_col] >= zmin) & (df_in[depth_col] <= zmax)].copy()


def safe_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return np.nanstd(x, ddof=1)


def load_zenith_from_mat(mat_path, struct_name="Zenith_CS", field_name="Zenith"):
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
        zenith_list.append(np.nan if value.size == 0 else float(value))

    return np.asarray(zenith_list, dtype=float)


def build_zenith_mapping(profile_ids_sorted, zenith_deg_arr, profile_id_order_in_mat=None):
    profile_ids_sorted = list(profile_ids_sorted)
    zenith_deg_arr = np.asarray(zenith_deg_arr, dtype=float)

    if profile_id_order_in_mat is None:
        if len(profile_ids_sorted) != len(zenith_deg_arr):
            raise ValueError(
                f"sorted(profile_id) 数量={len(profile_ids_sorted)}，"
                f"但 zenith 数量={len(zenith_deg_arr)}，无法默认一一对应。"
            )
        order = profile_ids_sorted
    else:
        order = list(profile_id_order_in_mat)
        if len(order) != len(zenith_deg_arr):
            raise ValueError(
                f"profile_id_order_in_mat 数量={len(order)}，"
                f"但 zenith 数量={len(zenith_deg_arr)}，不一致。"
            )

    return dict(zip(order, zenith_deg_arr))


def build_feature_matrix(profile_df, model_mode, chl_transform_spec, surface_kd_transform_mode, zenith_transform_mode):
    chl_used = transform_chl(profile_df["top_chl"].to_numpy(dtype=float), chl_transform_spec)
    surface_kd_used = transform_surface_kd(profile_df["surface_kd"].to_numpy(dtype=float), surface_kd_transform_mode)
    zenith_used = transform_zenith(profile_df["zenith_deg"].to_numpy(dtype=float), zenith_transform_mode)

    if model_mode == "chl_only":
        X = np.column_stack([chl_used])
    elif model_mode == "surfacekd_only":
        X = np.column_stack([surface_kd_used])
    elif model_mode == "zenith_only":
        X = np.column_stack([zenith_used])
    elif model_mode == "chl_zenith":
        X = np.column_stack([chl_used, zenith_used])
    elif model_mode == "surfacekd_zenith":
        X = np.column_stack([surface_kd_used, zenith_used])
    elif model_mode == "chl_surfacekd":
        X = np.column_stack([chl_used, surface_kd_used])
    elif model_mode == "chl_surfacekd_zenith":
        X = np.column_stack([chl_used, surface_kd_used, zenith_used])
    else:
        raise ValueError(f"未知 model_mode: {model_mode}")

    return X, chl_used, surface_kd_used, zenith_used


def fit_profile_a_c(profile_one):
    x = profile_one[baseline_col].to_numpy(dtype=float)
    y = profile_one[y_col].to_numpy(dtype=float)

    valid = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    A = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a_hat = float(coef[0])
    c_hat = float(coef[1])

    y_fit = a_hat * x + c_hat
    rmse_fit = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    return a_hat, c_hat, rmse_fit


def reconstruct_profile_with_ac(profile_one, a_pred, c_pred):
    b = profile_one[baseline_col].to_numpy(dtype=float)
    return a_pred * b + c_pred


def clip_a_values(a):
    if not clip_a_pred:
        return a
    return np.clip(a, clip_a_min, clip_a_max)


def build_profile_level_df(df_range, target_col_name, target_source_col):
    profile_df = (
        df_range.groupby(profile_col)
        .agg(
            top_chl=(chl_col, "first"),
            target_true=(target_source_col, "mean"),
            target_std=(target_source_col, safe_std),
            n_depth_points=(target_source_col, "size"),
            depth_min=(depth_col, "min"),
            depth_max=(depth_col, "max"),
            depth_mean=(depth_col, "mean"),
        )
        .reset_index()
    )

    if min_points_per_profile_in_range is not None and min_points_per_profile_in_range > 1:
        profile_df = profile_df[profile_df["n_depth_points"] >= min_points_per_profile_in_range].copy()

    profile_df = profile_df.merge(surface_kd_df, on=profile_col, how="left")
    profile_df["zenith_deg"] = profile_df[profile_col].map(zenith_mapping)
    profile_df = profile_df.rename(columns={"target_true": target_col_name})
    return profile_df


# =========================================================
# 6. 读取逐深度数据
# =========================================================

df = pd.read_csv(input_csv)
required_cols = [profile_col, chl_col, depth_col, kd_col, y_col, baseline_col, target_depthwise_col]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"输入文件缺少必要列: {missing_cols}")

df = df.dropna(subset=[profile_col, chl_col, depth_col, kd_col, y_col, baseline_col, target_depthwise_col]).copy()
if exclude_profile_ids:
    df = df[~df[profile_col].isin(exclude_profile_ids)].copy()

print("读取数据完成")
print("target_branch =", target_branch)
print("training_target_mode =", training_target_mode)
print("使用的 y_col =", y_col)
print("使用的 baseline_col =", baseline_col)
print("使用的 target_depthwise_col =", target_depthwise_col)
print("逐深度数据维度：", df.shape)
print("profile 数量：", df[profile_col].nunique())


# =========================================================
# 7. 提取 surface_kd（最浅层目标 kd）
# =========================================================

surface_kd_df = (
    df.sort_values([profile_col, depth_col])
    .groupby(profile_col, as_index=False)
    .first()[[profile_col, depth_col, kd_col]]
    .rename(columns={depth_col: "surface_depth", kd_col: "surface_kd"})
)

print("\n提取的 surface_kd：")
print(surface_kd_df.head())


# =========================================================
# 8. 读取 zenith.mat
# =========================================================

profile_ids_sorted = sorted(df[profile_col].unique().tolist())
zenith_deg_arr = load_zenith_from_mat(zenith_mat_path, struct_name="Zenith_CS", field_name="Zenith")
zenith_mapping = build_zenith_mapping(profile_ids_sorted, zenith_deg_arr, profile_id_order_in_mat)

print("\n已读取 zenith.mat")
print("天顶角数量：", len(zenith_deg_arr))


# =========================================================
# 9A. residual 模式：使用基线
# =========================================================

def run_one_depth_range_analysis_residual(df_all, depth_range):
    range_label = depth_range_to_label(depth_range)
    range_output_dir = os.path.join(output_dir, range_label)
    os.makedirs(range_output_dir, exist_ok=True)

    profile_target_csv = os.path.join(range_output_dir, f"profile_mean_b_for_gam_{target_branch}.csv")
    eval_csv = os.path.join(range_output_dir, f"gam_profile_b_gridsearch_eval_{target_branch}.csv")
    all_pred_csv = os.path.join(range_output_dir, f"all_gam_profile_b_predictions_{target_branch}.csv")
    best_pred_csv = os.path.join(range_output_dir, f"best_gam_profile_b_prediction_{target_branch}.csv")
    best_model_path = os.path.join(range_output_dir, f"best_gam_profile_b_model_{target_branch}.joblib")
    depthwise_recon_csv = os.path.join(range_output_dir, f"depthwise_reconstruction_from_b_{target_branch}.csv")

    df_range = filter_df_by_depth_range(df_all, depth_col, depth_range)
    print("\n" + "=" * 60)
    print(f"[residual] 当前分析范围：{depth_range}")
    print("该范围逐深度数据维度：", df_range.shape)

    if df_range.empty:
        print("该深度范围内没有数据，跳过。")
        return None

    profile_df = build_profile_level_df(df_range, "target_true", target_depthwise_col)

    if remove_outlier_profiles and len(profile_df) >= 3:
        std_val = profile_df["target_true"].std()
        if pd.notna(std_val) and std_val > 0:
            z_score = (profile_df["target_true"] - profile_df["target_true"].mean()) / std_val
            profile_df = profile_df[np.abs(z_score) <= outlier_z_threshold].copy()

    if len(profile_df) < 3:
        print("该范围内可用于建模的 profile 太少，跳过。")
        return None

    profile_df["depth_range_label"] = range_label
    profile_df["target_branch"] = target_branch
    profile_df["training_target_mode"] = "residual"
    profile_df["depth_range_min"] = np.nan if depth_range is None else depth_range[0]
    profile_df["depth_range_max"] = np.nan if depth_range is None else depth_range[1]
    profile_df.to_csv(profile_target_csv, index=False, encoding="utf-8-sig")

    y = profile_df["target_true"].to_numpy(dtype=float)
    eval_records = []
    pred_records = []

    total_count = (
        len(model_mode_candidates)
        * len(chl_transform_candidates)
        * len(surface_kd_transform_candidates)
        * len(zenith_transform_candidates)
        * len(n_splines_candidates)
        * len(lam_candidates)
    )
    now_count = 0

    for model_mode in model_mode_candidates:
        for chl_transform_spec in chl_transform_candidates:
            for surface_kd_transform_mode in surface_kd_transform_candidates:
                for zenith_transform_mode in zenith_transform_candidates:
                    X, chl_used, surface_kd_used, zenith_used = build_feature_matrix(
                        profile_df, model_mode, chl_transform_spec, surface_kd_transform_mode, zenith_transform_mode
                    )
                    for n_splines in n_splines_candidates:
                        for lam in lam_candidates:
                            now_count += 1
                            y_pred = loocv_predict_gam(X, y, n_splines, lam, use_interaction)
                            metrics = calc_metrics(y, y_pred)

                            eval_records.append({
                                "target_branch": target_branch,
                                "training_target_mode": "residual",
                                "depth_range_label": range_label,
                                "depth_range_min": np.nan if depth_range is None else depth_range[0],
                                "depth_range_max": np.nan if depth_range is None else depth_range[1],
                                "model_mode": model_mode,
                                "chl_transform": chl_transform_spec["name"],
                                "chl_transform_kind": chl_transform_spec["kind"],
                                "chl_power": chl_transform_spec["power"],
                                "surface_kd_transform": surface_kd_transform_mode,
                                "zenith_transform": zenith_transform_mode,
                                "n_splines": n_splines,
                                "lam": lam,
                                "loocv_corr": metrics["corr"],
                                "loocv_mae": metrics["mae"],
                                "loocv_rmse": metrics["rmse"],
                                "loocv_r2": metrics["r2"],
                            })

                            tmp = profile_df.copy()
                            tmp["model_mode"] = model_mode
                            tmp["chl_transform"] = chl_transform_spec["name"]
                            tmp["chl_transform_kind"] = chl_transform_spec["kind"]
                            tmp["chl_power"] = chl_transform_spec["power"]
                            tmp["surface_kd_transform"] = surface_kd_transform_mode
                            tmp["zenith_transform"] = zenith_transform_mode
                            tmp["top_chl_used"] = chl_used
                            tmp["surface_kd_used"] = surface_kd_used
                            tmp["zenith_used"] = zenith_used
                            tmp["n_splines"] = n_splines
                            tmp["lam"] = lam
                            tmp["target_pred"] = y_pred
                            tmp["target_error"] = tmp["target_pred"] - tmp["target_true"]
                            tmp["target_abs_error"] = tmp["target_error"].abs()
                            pred_records.append(tmp)

                            print(
                                f"[{now_count}/{total_count}] residual | {target_branch} | {range_label} | "
                                f"mode={model_mode} | chl={chl_transform_spec['name']}, skd={surface_kd_transform_mode}, "
                                f"zen={zenith_transform_mode}, s={n_splines}, lam={lam}, "
                                f"RMSE={metrics['rmse']:.6f}, Corr={metrics['corr']:.3f}"
                            )

    eval_df = pd.DataFrame(eval_records).sort_values("loocv_rmse", ascending=True).reset_index(drop=True)
    pred_df = pd.concat(pred_records, ignore_index=True)
    eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(all_pred_csv, index=False, encoding="utf-8-sig")

    best = eval_df.iloc[0]
    best_model_mode = best["model_mode"]
    best_chl_transform = best["chl_transform"]
    best_surface_kd_transform = best["surface_kd_transform"]
    best_zenith_transform = best["zenith_transform"]
    best_n_splines = int(best["n_splines"])
    best_lam = float(best["lam"])
    best_chl_transform_spec = find_transform_spec_by_name(best_chl_transform, chl_transform_candidates)

    best_pred_df = pred_df[
        (pred_df["model_mode"] == best_model_mode)
        & (pred_df["chl_transform"] == best_chl_transform)
        & (pred_df["surface_kd_transform"] == best_surface_kd_transform)
        & (pred_df["zenith_transform"] == best_zenith_transform)
        & (pred_df["n_splines"] == best_n_splines)
        & (np.isclose(pred_df["lam"].astype(float), best_lam))
    ].copy().sort_values(profile_col).reset_index(drop=True)
    best_pred_df.to_csv(best_pred_csv, index=False, encoding="utf-8-sig")

    X_best, _, _, _ = build_feature_matrix(
        profile_df, best_model_mode, best_chl_transform_spec, best_surface_kd_transform, best_zenith_transform
    )
    final_gam = build_gam(best_n_splines, best_lam, X_best.shape[1], use_interaction)
    final_gam.fit(X_best, y)

    joblib.dump(
        {
            "model": final_gam,
            "training_target_mode": "residual",
            "target_branch": target_branch,
            "target_kd_col": kd_col,
            "target_depthwise_col": target_depthwise_col,
            "y_col": y_col,
            "baseline_col": baseline_col,
            "depth_range_label": range_label,
            "depth_range_min": None if depth_range is None else depth_range[0],
            "depth_range_max": None if depth_range is None else depth_range[1],
            "model_mode": best_model_mode,
            "chl_transform": best_chl_transform,
            "chl_transform_kind": best_chl_transform_spec["kind"],
            "chl_power": best_chl_transform_spec["power"],
            "surface_kd_transform": best_surface_kd_transform,
            "zenith_transform": best_zenith_transform,
            "n_splines": best_n_splines,
            "lam": best_lam,
            "input": best_model_mode,
            "target": f"profile_mean_residual_in_depth_range_{target_branch}",
        },
        best_model_path,
    )

    if save_depthwise_reconstruction_csv:
        best_map = best_pred_df[[profile_col, "target_pred"]].rename(columns={"target_pred": "pred_range_residual"})
        recon_df = df_range.merge(best_map, on=profile_col, how="left")
        recon_df["y_reconstructed"] = recon_df[baseline_col] + recon_df["pred_range_residual"]
        recon_df["recon_error"] = recon_df["y_reconstructed"] - recon_df[y_col]
        recon_df.to_csv(depthwise_recon_csv, index=False, encoding="utf-8-sig")

    if make_plot:
        best_pred_profile_df = (
            best_pred_df.groupby(profile_col, as_index=False)
            .agg(
                top_chl=("top_chl", "first"),
                surface_kd=("surface_kd", "first"),
                zenith_deg=("zenith_deg", "first"),
                target_true=("target_true", "mean"),
                target_pred=("target_pred", "mean"),
            )
            .sort_values(profile_col)
            .reset_index(drop=True)
        )

        plt.figure(figsize=(6, 6))
        plt.scatter(best_pred_profile_df["target_true"], best_pred_profile_df["target_pred"], s=50)
        for _, row in best_pred_profile_df.iterrows():
            plt.text(row["target_true"], row["target_pred"], str(int(row[profile_col])), fontsize=8)
        lo = min(best_pred_profile_df["target_true"].min(), best_pred_profile_df["target_pred"].min())
        hi = max(best_pred_profile_df["target_true"].max(), best_pred_profile_df["target_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", linewidth=2)
        plt.xlabel("Observed mean residual")
        plt.ylabel("Predicted mean residual")
        plt.title(f"residual | {target_branch} | {range_label}, r={calc_corr(best_pred_profile_df['target_true'], best_pred_profile_df['target_pred']):.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(range_output_dir, f"scatter_residual_true_vs_pred_{target_branch}.png"), dpi=300, bbox_inches="tight")
        plt.show()

    return {
        "training_target_mode": "residual",
        "target_branch": target_branch,
        "depth_range_label": range_label,
        "depth_range_min": np.nan if depth_range is None else depth_range[0],
        "depth_range_max": np.nan if depth_range is None else depth_range[1],
        "n_profiles": len(profile_df),
        "best_model_mode": best_model_mode,
        "best_chl_transform": best_chl_transform,
        "best_chl_power": float(best["chl_power"]) if pd.notna(best["chl_power"]) else np.nan,
        "best_surface_kd_transform": best_surface_kd_transform,
        "best_zenith_transform": best_zenith_transform,
        "best_n_splines": best_n_splines,
        "best_lam": best_lam,
        "best_rmse": float(best["loocv_rmse"]),
        "best_corr": float(best["loocv_corr"]) if pd.notna(best["loocv_corr"]) else np.nan,
        "best_r2": float(best["loocv_r2"]) if pd.notna(best["loocv_r2"]) else np.nan,
        "output_dir": range_output_dir,
    }


# =========================================================
# 9B. direct_y 模式：不使用基线
# =========================================================

def run_one_depth_range_analysis_direct_y(df_all, depth_range):
    range_label = depth_range_to_label(depth_range)
    range_output_dir = os.path.join(output_dir, range_label)
    os.makedirs(range_output_dir, exist_ok=True)

    profile_target_csv = os.path.join(range_output_dir, f"profile_mean_y_for_gam_{target_branch}.csv")
    eval_csv = os.path.join(range_output_dir, f"gam_profile_y_gridsearch_eval_{target_branch}.csv")
    all_pred_csv = os.path.join(range_output_dir, f"all_gam_profile_y_predictions_{target_branch}.csv")
    best_pred_csv = os.path.join(range_output_dir, f"best_gam_profile_y_prediction_{target_branch}.csv")
    best_model_path = os.path.join(range_output_dir, f"best_gam_profile_y_model_{target_branch}.joblib")
    depthwise_recon_csv = os.path.join(range_output_dir, f"depthwise_reconstruction_from_y_{target_branch}.csv")

    df_range = filter_df_by_depth_range(df_all, depth_col, depth_range)
    print("\n" + "=" * 60)
    print(f"[direct_y] 当前分析范围：{depth_range}")
    print("该范围逐深度数据维度：", df_range.shape)

    if df_range.empty:
        print("该深度范围内没有数据，跳过。")
        return None

    profile_df = build_profile_level_df(df_range, "target_true", y_col)

    if remove_outlier_profiles and len(profile_df) >= 3:
        std_val = profile_df["target_true"].std()
        if pd.notna(std_val) and std_val > 0:
            z_score = (profile_df["target_true"] - profile_df["target_true"].mean()) / std_val
            profile_df = profile_df[np.abs(z_score) <= outlier_z_threshold].copy()

    if len(profile_df) < 3:
        print("该范围内可用于建模的 profile 太少，跳过。")
        return None

    profile_df["depth_range_label"] = range_label
    profile_df["target_branch"] = target_branch
    profile_df["training_target_mode"] = "direct_y"
    profile_df["depth_range_min"] = np.nan if depth_range is None else depth_range[0]
    profile_df["depth_range_max"] = np.nan if depth_range is None else depth_range[1]
    profile_df.to_csv(profile_target_csv, index=False, encoding="utf-8-sig")

    y = profile_df["target_true"].to_numpy(dtype=float)
    eval_records = []
    pred_records = []

    total_count = (
        len(model_mode_candidates)
        * len(chl_transform_candidates)
        * len(surface_kd_transform_candidates)
        * len(zenith_transform_candidates)
        * len(n_splines_candidates)
        * len(lam_candidates)
    )
    now_count = 0

    for model_mode in model_mode_candidates:
        for chl_transform_spec in chl_transform_candidates:
            for surface_kd_transform_mode in surface_kd_transform_candidates:
                for zenith_transform_mode in zenith_transform_candidates:
                    X, chl_used, surface_kd_used, zenith_used = build_feature_matrix(
                        profile_df, model_mode, chl_transform_spec, surface_kd_transform_mode, zenith_transform_mode
                    )
                    for n_splines in n_splines_candidates:
                        for lam in lam_candidates:
                            now_count += 1
                            y_pred = loocv_predict_gam(X, y, n_splines, lam, use_interaction)
                            metrics = calc_metrics(y, y_pred)

                            eval_records.append({
                                "target_branch": target_branch,
                                "training_target_mode": "direct_y",
                                "depth_range_label": range_label,
                                "depth_range_min": np.nan if depth_range is None else depth_range[0],
                                "depth_range_max": np.nan if depth_range is None else depth_range[1],
                                "model_mode": model_mode,
                                "chl_transform": chl_transform_spec["name"],
                                "chl_transform_kind": chl_transform_spec["kind"],
                                "chl_power": chl_transform_spec["power"],
                                "surface_kd_transform": surface_kd_transform_mode,
                                "zenith_transform": zenith_transform_mode,
                                "n_splines": n_splines,
                                "lam": lam,
                                "loocv_corr": metrics["corr"],
                                "loocv_mae": metrics["mae"],
                                "loocv_rmse": metrics["rmse"],
                                "loocv_r2": metrics["r2"],
                            })

                            tmp = profile_df.copy()
                            tmp["model_mode"] = model_mode
                            tmp["chl_transform"] = chl_transform_spec["name"]
                            tmp["chl_transform_kind"] = chl_transform_spec["kind"]
                            tmp["chl_power"] = chl_transform_spec["power"]
                            tmp["surface_kd_transform"] = surface_kd_transform_mode
                            tmp["zenith_transform"] = zenith_transform_mode
                            tmp["top_chl_used"] = chl_used
                            tmp["surface_kd_used"] = surface_kd_used
                            tmp["zenith_used"] = zenith_used
                            tmp["n_splines"] = n_splines
                            tmp["lam"] = lam
                            tmp["target_pred"] = y_pred
                            tmp["target_error"] = tmp["target_pred"] - tmp["target_true"]
                            tmp["target_abs_error"] = tmp["target_error"].abs()
                            pred_records.append(tmp)

                            print(
                                f"[{now_count}/{total_count}] direct_y | {target_branch} | {range_label} | "
                                f"mode={model_mode} | chl={chl_transform_spec['name']}, skd={surface_kd_transform_mode}, "
                                f"zen={zenith_transform_mode}, s={n_splines}, lam={lam}, "
                                f"RMSE={metrics['rmse']:.6f}, Corr={metrics['corr']:.3f}"
                            )

    eval_df = pd.DataFrame(eval_records).sort_values("loocv_rmse", ascending=True).reset_index(drop=True)
    pred_df = pd.concat(pred_records, ignore_index=True)
    eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(all_pred_csv, index=False, encoding="utf-8-sig")

    best = eval_df.iloc[0]
    best_model_mode = best["model_mode"]
    best_chl_transform = best["chl_transform"]
    best_surface_kd_transform = best["surface_kd_transform"]
    best_zenith_transform = best["zenith_transform"]
    best_n_splines = int(best["n_splines"])
    best_lam = float(best["lam"])
    best_chl_transform_spec = find_transform_spec_by_name(best_chl_transform, chl_transform_candidates)

    best_pred_df = pred_df[
        (pred_df["model_mode"] == best_model_mode)
        & (pred_df["chl_transform"] == best_chl_transform)
        & (pred_df["surface_kd_transform"] == best_surface_kd_transform)
        & (pred_df["zenith_transform"] == best_zenith_transform)
        & (pred_df["n_splines"] == best_n_splines)
        & (np.isclose(pred_df["lam"].astype(float), best_lam))
    ].copy().sort_values(profile_col).reset_index(drop=True)
    best_pred_df.to_csv(best_pred_csv, index=False, encoding="utf-8-sig")

    X_best, _, _, _ = build_feature_matrix(
        profile_df, best_model_mode, best_chl_transform_spec, best_surface_kd_transform, best_zenith_transform
    )
    final_gam = build_gam(best_n_splines, best_lam, X_best.shape[1], use_interaction)
    final_gam.fit(X_best, y)

    joblib.dump(
        {
            "model": final_gam,
            "training_target_mode": "direct_y",
            "target_branch": target_branch,
            "target_kd_col": kd_col,
            "target_depthwise_col": target_depthwise_col,
            "y_col": y_col,
            "baseline_col": baseline_col,
            "depth_range_label": range_label,
            "depth_range_min": None if depth_range is None else depth_range[0],
            "depth_range_max": None if depth_range is None else depth_range[1],
            "model_mode": best_model_mode,
            "chl_transform": best_chl_transform,
            "chl_transform_kind": best_chl_transform_spec["kind"],
            "chl_power": best_chl_transform_spec["power"],
            "surface_kd_transform": best_surface_kd_transform,
            "zenith_transform": best_zenith_transform,
            "n_splines": best_n_splines,
            "lam": best_lam,
            "input": best_model_mode,
            "target": f"profile_mean_y_in_depth_range_{target_branch}",
        },
        best_model_path,
    )

    if save_depthwise_reconstruction_csv:
        best_map = best_pred_df[[profile_col, "target_pred"]].rename(columns={"target_pred": "pred_range_y"})
        recon_df = df_range.merge(best_map, on=profile_col, how="left")
        recon_df["y_reconstructed"] = recon_df["pred_range_y"]
        recon_df["recon_error"] = recon_df["y_reconstructed"] - recon_df[y_col]
        recon_df.to_csv(depthwise_recon_csv, index=False, encoding="utf-8-sig")

    if make_plot:
        best_pred_profile_df = (
            best_pred_df.groupby(profile_col, as_index=False)
            .agg(
                top_chl=("top_chl", "first"),
                surface_kd=("surface_kd", "first"),
                zenith_deg=("zenith_deg", "first"),
                target_true=("target_true", "mean"),
                target_pred=("target_pred", "mean"),
            )
            .sort_values(profile_col)
            .reset_index(drop=True)
        )

        plt.figure(figsize=(6, 6))
        plt.scatter(best_pred_profile_df["target_true"], best_pred_profile_df["target_pred"], s=50)
        for _, row in best_pred_profile_df.iterrows():
            plt.text(row["target_true"], row["target_pred"], str(int(row[profile_col])), fontsize=8)
        lo = min(best_pred_profile_df["target_true"].min(), best_pred_profile_df["target_pred"].min())
        hi = max(best_pred_profile_df["target_true"].max(), best_pred_profile_df["target_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", linewidth=2)
        plt.xlabel("Observed mean Y")
        plt.ylabel("Predicted mean Y")
        plt.title(f"direct_y | {target_branch} | {range_label}, r={calc_corr(best_pred_profile_df['target_true'], best_pred_profile_df['target_pred']):.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(range_output_dir, f"scatter_y_true_vs_pred_{target_branch}.png"), dpi=300, bbox_inches="tight")
        plt.show()

    return {
        "training_target_mode": "direct_y",
        "target_branch": target_branch,
        "depth_range_label": range_label,
        "depth_range_min": np.nan if depth_range is None else depth_range[0],
        "depth_range_max": np.nan if depth_range is None else depth_range[1],
        "n_profiles": len(profile_df),
        "best_model_mode": best_model_mode,
        "best_chl_transform": best_chl_transform,
        "best_chl_power": float(best["chl_power"]) if pd.notna(best["chl_power"]) else np.nan,
        "best_surface_kd_transform": best_surface_kd_transform,
        "best_zenith_transform": best_zenith_transform,
        "best_n_splines": best_n_splines,
        "best_lam": best_lam,
        "best_rmse": float(best["loocv_rmse"]),
        "best_corr": float(best["loocv_corr"]) if pd.notna(best["loocv_corr"]) else np.nan,
        "best_r2": float(best["loocv_r2"]) if pd.notna(best["loocv_r2"]) else np.nan,
        "output_dir": range_output_dir,
    }


# =========================================================
# 9C. ac 模式：使用基线，但训练 a/c
# =========================================================

def run_one_depth_range_analysis_ac(df_all, depth_range):
    range_label = depth_range_to_label(depth_range)
    range_output_dir = os.path.join(output_dir, range_label)
    os.makedirs(range_output_dir, exist_ok=True)

    profile_param_csv = os.path.join(range_output_dir, f"profile_ac_for_gam_{target_branch}.csv")
    eval_csv = os.path.join(range_output_dir, f"gam_profile_ac_gridsearch_eval_{target_branch}.csv")
    all_pred_csv = os.path.join(range_output_dir, f"all_gam_profile_ac_predictions_{target_branch}.csv")
    best_pred_csv = os.path.join(range_output_dir, f"best_gam_profile_ac_prediction_{target_branch}.csv")
    best_model_path = os.path.join(range_output_dir, f"best_gam_profile_ac_model_{target_branch}.joblib")
    recon_profile_csv = os.path.join(range_output_dir, f"best_profile_reconstruction_{target_branch}.csv")

    df_range = filter_df_by_depth_range(df_all, depth_col, depth_range)
    print("\n" + "=" * 60)
    print(f"[ac] 当前分析范围：{depth_range}")
    print("该范围逐深度数据维度：", df_range.shape)

    if df_range.empty:
        print("该深度范围内没有数据，跳过。")
        return None

    profile_records = []
    for pid, g in df_range.groupby(profile_col):
        g = g.sort_values(depth_col).copy()
        if len(g) < min_points_per_profile_in_range:
            continue

        a_hat, c_hat, fit_rmse = fit_profile_a_c(g)
        if np.isnan(a_hat) or np.isnan(c_hat):
            continue

        profile_records.append({
            profile_col: pid,
            "top_chl": float(g[chl_col].iloc[0]),
            "a_true": a_hat,
            "c_true": c_hat,
            "fit_rmse": fit_rmse,
            "n_depth_points": len(g),
            "depth_min": float(g[depth_col].min()),
            "depth_max": float(g[depth_col].max()),
            "depth_mean": float(g[depth_col].mean()),
        })

    profile_df = pd.DataFrame(profile_records)
    if profile_df.empty or len(profile_df) < 3:
        print("该范围内可用于建模的 profile 太少，跳过。")
        return None

    profile_df = profile_df.merge(surface_kd_df, on=profile_col, how="left")
    profile_df["zenith_deg"] = profile_df[profile_col].map(zenith_mapping)

    if remove_outlier_profiles and len(profile_df) >= 3:
        for col in ["a_true", "c_true"]:
            std_val = profile_df[col].std()
            if pd.notna(std_val) and std_val > 0:
                z_score = (profile_df[col] - profile_df[col].mean()) / std_val
                profile_df = profile_df[np.abs(z_score) <= outlier_z_threshold].copy()

    profile_df["depth_range_label"] = range_label
    profile_df["target_branch"] = target_branch
    profile_df["training_target_mode"] = "ac"
    profile_df["depth_range_min"] = np.nan if depth_range is None else depth_range[0]
    profile_df["depth_range_max"] = np.nan if depth_range is None else depth_range[1]
    profile_df.to_csv(profile_param_csv, index=False, encoding="utf-8-sig")

    y_a = profile_df["a_true"].to_numpy(dtype=float)
    y_c = profile_df["c_true"].to_numpy(dtype=float)
    eval_records = []
    pred_records = []

    total_count = (
        len(model_mode_candidates)
        * len(chl_transform_candidates)
        * len(surface_kd_transform_candidates)
        * len(zenith_transform_candidates)
        * len(n_splines_candidates)
        * len(lam_candidates)
    )
    now_count = 0

    for model_mode in model_mode_candidates:
        for chl_transform_spec in chl_transform_candidates:
            for surface_kd_transform_mode in surface_kd_transform_candidates:
                for zenith_transform_mode in zenith_transform_candidates:
                    X, chl_used, surface_kd_used, zenith_used = build_feature_matrix(
                        profile_df, model_mode, chl_transform_spec, surface_kd_transform_mode, zenith_transform_mode
                    )
                    for n_splines in n_splines_candidates:
                        for lam in lam_candidates:
                            now_count += 1
                            a_pred = loocv_predict_gam(X, y_a, n_splines, lam, use_interaction)
                            c_pred = loocv_predict_gam(X, y_c, n_splines, lam, use_interaction)
                            a_pred = clip_a_values(a_pred)

                            metrics_a = calc_metrics(y_a, a_pred)
                            metrics_c = calc_metrics(y_c, c_pred)
                            combined_score = metrics_a["rmse"] + metrics_c["rmse"]

                            eval_records.append({
                                "target_branch": target_branch,
                                "training_target_mode": "ac",
                                "depth_range_label": range_label,
                                "depth_range_min": np.nan if depth_range is None else depth_range[0],
                                "depth_range_max": np.nan if depth_range is None else depth_range[1],
                                "model_mode": model_mode,
                                "chl_transform": chl_transform_spec["name"],
                                "chl_transform_kind": chl_transform_spec["kind"],
                                "chl_power": chl_transform_spec["power"],
                                "surface_kd_transform": surface_kd_transform_mode,
                                "zenith_transform": zenith_transform_mode,
                                "n_splines": n_splines,
                                "lam": lam,
                                "a_corr": metrics_a["corr"],
                                "a_rmse": metrics_a["rmse"],
                                "a_r2": metrics_a["r2"],
                                "c_corr": metrics_c["corr"],
                                "c_rmse": metrics_c["rmse"],
                                "c_r2": metrics_c["r2"],
                                "combined_score": combined_score,
                            })

                            tmp = profile_df.copy()
                            tmp["model_mode"] = model_mode
                            tmp["chl_transform"] = chl_transform_spec["name"]
                            tmp["chl_transform_kind"] = chl_transform_spec["kind"]
                            tmp["chl_power"] = chl_transform_spec["power"]
                            tmp["surface_kd_transform"] = surface_kd_transform_mode
                            tmp["zenith_transform"] = zenith_transform_mode
                            tmp["top_chl_used"] = chl_used
                            tmp["surface_kd_used"] = surface_kd_used
                            tmp["zenith_used"] = zenith_used
                            tmp["n_splines"] = n_splines
                            tmp["lam"] = lam
                            tmp["a_pred"] = a_pred
                            tmp["c_pred"] = c_pred
                            tmp["a_error"] = tmp["a_pred"] - tmp["a_true"]
                            tmp["c_error"] = tmp["c_pred"] - tmp["c_true"]
                            tmp["a_abs_error"] = tmp["a_error"].abs()
                            tmp["c_abs_error"] = tmp["c_error"].abs()
                            pred_records.append(tmp)

                            print(
                                f"[{now_count}/{total_count}] ac | {target_branch} | {range_label} | "
                                f"mode={model_mode} | chl={chl_transform_spec['name']}, skd={surface_kd_transform_mode}, "
                                f"zen={zenith_transform_mode}, s={n_splines}, lam={lam}, "
                                f"a_RMSE={metrics_a['rmse']:.6f}, c_RMSE={metrics_c['rmse']:.6f}, score={combined_score:.6f}"
                            )

    eval_df = pd.DataFrame(eval_records).sort_values("combined_score", ascending=True).reset_index(drop=True)
    pred_df = pd.concat(pred_records, ignore_index=True)
    eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(all_pred_csv, index=False, encoding="utf-8-sig")

    best = eval_df.iloc[0]
    best_model_mode = best["model_mode"]
    best_chl_transform = best["chl_transform"]
    best_surface_kd_transform = best["surface_kd_transform"]
    best_zenith_transform = best["zenith_transform"]
    best_n_splines = int(best["n_splines"])
    best_lam = float(best["lam"])
    best_chl_transform_spec = find_transform_spec_by_name(best_chl_transform, chl_transform_candidates)

    best_pred_df = pred_df[
        (pred_df["model_mode"] == best_model_mode)
        & (pred_df["chl_transform"] == best_chl_transform)
        & (pred_df["surface_kd_transform"] == best_surface_kd_transform)
        & (pred_df["zenith_transform"] == best_zenith_transform)
        & (pred_df["n_splines"] == best_n_splines)
        & (np.isclose(pred_df["lam"].astype(float), best_lam))
    ].copy().sort_values(profile_col).reset_index(drop=True)
    best_pred_df.to_csv(best_pred_csv, index=False, encoding="utf-8-sig")

    X_best, _, _, _ = build_feature_matrix(
        profile_df, best_model_mode, best_chl_transform_spec, best_surface_kd_transform, best_zenith_transform
    )
    final_gam_a = build_gam(best_n_splines, best_lam, X_best.shape[1], use_interaction)
    final_gam_c = build_gam(best_n_splines, best_lam, X_best.shape[1], use_interaction)
    final_gam_a.fit(X_best, y_a)
    final_gam_c.fit(X_best, y_c)

    joblib.dump(
        {
            "model_a": final_gam_a,
            "model_c": final_gam_c,
            "training_target_mode": "ac",
            "target_branch": target_branch,
            "target_kd_col": kd_col,
            "target_depthwise_col": target_depthwise_col,
            "y_col": y_col,
            "baseline_col": baseline_col,
            "depth_range_label": range_label,
            "depth_range_min": None if depth_range is None else depth_range[0],
            "depth_range_max": None if depth_range is None else depth_range[1],
            "model_mode": best_model_mode,
            "chl_transform": best_chl_transform,
            "chl_transform_kind": best_chl_transform_spec["kind"],
            "chl_power": best_chl_transform_spec["power"],
            "surface_kd_transform": best_surface_kd_transform,
            "zenith_transform": best_zenith_transform,
            "n_splines": best_n_splines,
            "lam": best_lam,
            "clip_a_pred": clip_a_pred,
            "clip_a_min": clip_a_min,
            "clip_a_max": clip_a_max,
            "input": best_model_mode,
            "target": f"profile_a_c_in_depth_range_{target_branch}",
        },
        best_model_path,
    )

    if save_depthwise_reconstruction_csv:
        recon_records = []
        best_map = best_pred_df[[profile_col, "a_true", "c_true", "a_pred", "c_pred"]].copy()
        for pid, g in df_range.groupby(profile_col):
            g = g.sort_values(depth_col).copy()
            row = best_map[best_map[profile_col] == pid]
            if row.empty:
                continue
            row = row.iloc[0]

            a_pred = clip_a_values(float(row["a_pred"]))
            c_pred = float(row["c_pred"])
            y_pred_profile = reconstruct_profile_with_ac(g, a_pred, c_pred)

            tmp = g.copy()
            tmp["a_true"] = float(row["a_true"])
            tmp["c_true"] = float(row["c_true"])
            tmp["a_pred"] = a_pred
            tmp["c_pred"] = c_pred
            tmp["y_reconstructed"] = y_pred_profile
            tmp["recon_error"] = tmp["y_reconstructed"] - tmp[y_col]
            recon_records.append(tmp)

        recon_df = pd.concat(recon_records, ignore_index=True)
        recon_df.to_csv(recon_profile_csv, index=False, encoding="utf-8-sig")

    if make_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(best_pred_df["a_true"], best_pred_df["a_pred"], s=50)
        for _, row in best_pred_df.iterrows():
            plt.text(row["a_true"], row["a_pred"], str(int(row[profile_col])), fontsize=8)
        lo = min(best_pred_df["a_true"].min(), best_pred_df["a_pred"].min())
        hi = max(best_pred_df["a_true"].max(), best_pred_df["a_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", linewidth=2)
        plt.xlabel("Observed a")
        plt.ylabel("Predicted a")
        plt.title(f"ac | {target_branch} | {range_label}: a, r={calc_corr(best_pred_df['a_true'], best_pred_df['a_pred']):.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(range_output_dir, f"scatter_a_true_vs_pred_{target_branch}.png"), dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(best_pred_df["c_true"], best_pred_df["c_pred"], s=50)
        for _, row in best_pred_df.iterrows():
            plt.text(row["c_true"], row["c_pred"], str(int(row[profile_col])), fontsize=8)
        lo = min(best_pred_df["c_true"].min(), best_pred_df["c_pred"].min())
        hi = max(best_pred_df["c_true"].max(), best_pred_df["c_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", linewidth=2)
        plt.xlabel("Observed c")
        plt.ylabel("Predicted c")
        plt.title(f"ac | {target_branch} | {range_label}: c, r={calc_corr(best_pred_df['c_true'], best_pred_df['c_pred']):.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(range_output_dir, f"scatter_c_true_vs_pred_{target_branch}.png"), dpi=300, bbox_inches="tight")
        plt.show()

    return {
        "training_target_mode": "ac",
        "target_branch": target_branch,
        "depth_range_label": range_label,
        "depth_range_min": np.nan if depth_range is None else depth_range[0],
        "depth_range_max": np.nan if depth_range is None else depth_range[1],
        "n_profiles": len(profile_df),
        "best_model_mode": best_model_mode,
        "best_chl_transform": best_chl_transform,
        "best_chl_power": float(best["chl_power"]) if pd.notna(best["chl_power"]) else np.nan,
        "best_surface_kd_transform": best_surface_kd_transform,
        "best_zenith_transform": best_zenith_transform,
        "best_n_splines": best_n_splines,
        "best_lam": best_lam,
        "best_a_rmse": float(best["a_rmse"]),
        "best_a_corr": float(best["a_corr"]) if pd.notna(best["a_corr"]) else np.nan,
        "best_c_rmse": float(best["c_rmse"]),
        "best_c_corr": float(best["c_corr"]) if pd.notna(best["c_corr"]) else np.nan,
        "best_combined_score": float(best["combined_score"]),
        "output_dir": range_output_dir,
    }


# =========================================================
# 10. 主调度
# =========================================================

depth_ranges = normalize_depth_ranges(depth_ranges)
summary_records = []

for one_range in depth_ranges:
    if training_target_mode == "residual":
        result = run_one_depth_range_analysis_residual(df, one_range)
    elif training_target_mode == "direct_y":
        result = run_one_depth_range_analysis_direct_y(df, one_range)
    elif training_target_mode == "ac":
        result = run_one_depth_range_analysis_ac(df, one_range)
    else:
        raise ValueError("training_target_mode 只能是 'residual' / 'direct_y' / 'ac'")

    if result is not None:
        summary_records.append(result)

if len(summary_records) > 0:
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_dir, f"depth_range_training_summary_{training_target_mode}_{target_branch}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n各深度范围汇总结果：")
    print(summary_df)
    print("已保存：", summary_csv)
else:
    print("\n没有任何深度范围成功完成分析。")

print("\n全部完成。输出目录：")
print(output_dir)
