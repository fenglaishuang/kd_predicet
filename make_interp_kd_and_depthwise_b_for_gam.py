import os
import numpy as np
import pandas as pd


# =========================================================
# make_interp_kd_and_depthwise_b_for_gam.py
#
# 功能：
# 1. 读取 train_df_step7.csv
# 2. 将每条 profile 的 Kd_slow / Kd_fast / chl / temperature / salinity
#    插值到统一深度网格：0~28 m，间隔 0.5 m
# 3. 计算每个深度上的基线：
#       B_y_slow / B_y_fast / B_chl / B_temp / B_sal / B_density
# 4. 计算逐深度偏移：
#       b_slow_i(z) = Y_slow_i(z) - B_y_slow(z)
#       b_fast_i(z) = Y_fast_i(z) - B_y_fast(z)
# 5. 输出一个尽量完整的主表 master.csv
# 6. 输出一个用于 GAM 的 for_gam.csv（slow 和 fast 都在同一个表里）
#
# 说明：
# - 不做 SVD
# - 不把 b_i(z) 平均成每条 profile 一个 b_i
# - 密度考虑深度影响：rho = rho(S, T, P(depth))
# - 为兼容旧脚本，generic 别名列：
#       kd_interp / Y_interp / B_y / b_depthwise
#   默认仍映射到 slow
# =========================================================


# =========================================================
# 1. 路径设置
# =========================================================

input_csv = r"D:\work\train_df_step7.csv"

output_dir = r"D:\work\gam_b_depthwise_data"
os.makedirs(output_dir, exist_ok=True)

master_output_csv = os.path.join(output_dir, "interp_kd_chl_density_depthwise_b_master.csv")
train_output_csv = os.path.join(output_dir, "interp_kd_chl_density_depthwise_b_for_gam.csv")
baseline_output_csv = os.path.join(output_dir, "depth_baseline_all_vars.csv")


# =========================================================
# 2. 基础列名
# =========================================================

profile_col = "profile_id"
depth_col = "depth"

kd_slow_col = "Kd_slow"
kd_fast_col = "Kd_fast"   # 如果原始文件里没有这一列，会自动跳过 fast 分支

chl_col = "chl"
temp_col = "temperature"
sal_col = "salinity"


# =========================================================
# 3. 可调参数
# =========================================================

exclude_profile_ids = []   # 例如 [22]

depth_start = 0.0
depth_end = 80.0
depth_step = 0.5

# Kd 变换方式
# "raw"   : 直接使用 Kd
# "log1p" : 使用 log(1 + Kd)
# "sqrt"  : 使用 sqrt(Kd)
kd_transform = "raw"

# 基线计算方式
# "mean" / "median"
baseline_method = "mean"

# 基线平滑窗口
baseline_smooth_window = 1

# 超出 profile 原始深度范围时如何处理
# "nan"  : 超出范围填 NaN
# "edge" : 用边界值延拓
# "skip" : profile 不能覆盖完整网格则整条跳过
out_of_range_mode = "nan"

# 是否删除训练输出中的关键 NaN 行
drop_nan_for_training = True

# 是否删除完全没有有效 profile 的深度层
drop_depths_with_zero_valid_profiles = True

# 压强计算所需纬度
latitude_deg = 30.0

# 是否尝试处理 Kd_fast
enable_kd_fast_if_exists = True

# generic 别名列默认映射哪一套 Kd
# 可选 "slow" / "fast"
primary_kd_branch = "slow"


# =========================================================
# 4. 工具函数
# =========================================================

def transform_kd(kd, method="raw"):
    kd = np.asarray(kd, dtype=float)

    if method == "raw":
        return kd
    elif method == "log1p":
        if np.any(kd < -1):
            raise ValueError("Kd 中存在小于 -1 的值，不能使用 log1p")
        return np.log1p(kd)
    elif method == "sqrt":
        if np.any(kd < 0):
            raise ValueError("Kd 中存在负值，不能使用 sqrt")
        return np.sqrt(kd)
    else:
        raise ValueError(f"未知 kd_transform: {method}")


def interp_profile_to_grid(z, y, depth_grid, mode="nan"):
    """
    将单条 profile 的单个变量插值到统一深度网格。
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (~np.isnan(z)) & (~np.isnan(y))
    z = z[valid]
    y = y[valid]

    if len(z) < 2:
        return None if mode == "skip" else np.full(len(depth_grid), np.nan)

    order = np.argsort(z)
    z = z[order]
    y = y[order]

    # 同一深度重复值取平均
    tmp = pd.DataFrame({"z": z, "y": y})
    tmp = tmp.groupby("z", as_index=False)["y"].mean().sort_values("z")

    z = tmp["z"].to_numpy(dtype=float)
    y = tmp["y"].to_numpy(dtype=float)

    if len(z) < 2:
        return None if mode == "skip" else np.full(len(depth_grid), np.nan)

    z_min = z.min()
    z_max = z.max()

    if mode == "skip":
        if depth_grid.min() < z_min or depth_grid.max() > z_max:
            return None
        return np.interp(depth_grid, z, y)

    elif mode == "edge":
        return np.interp(depth_grid, z, y)

    elif mode == "nan":
        y_interp = np.interp(depth_grid, z, y)
        outside = (depth_grid < z_min) | (depth_grid > z_max)
        y_interp[outside] = np.nan
        return y_interp

    else:
        raise ValueError(f"未知 out_of_range_mode: {mode}")


def interp_or_nan(z, y, depth_grid, mode="nan"):
    """
    interp_profile_to_grid 的安全包装：
    如果返回 None，则统一替换成全 NaN 数组，便于后续矩阵拼接。
    """
    arr = interp_profile_to_grid(z, y, depth_grid, mode=mode)
    if arr is None:
        return np.full(len(depth_grid), np.nan)
    return arr


def smooth_1d(values, window=1):
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values.copy()

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def depth_to_pressure_dbar(depth_m, latitude_deg=30.0):
    """
    深度 -> 压强（dbar）
    Saunders 1981 常用近似。
    """
    z = np.asarray(depth_m, dtype=float)
    x = np.sin(np.deg2rad(latitude_deg)) ** 2
    c1 = (5.92 + 5.25 * x) * 1e-3
    p = ((1 - c1) - np.sqrt((1 - c1) ** 2 - 8.84e-6 * z)) / 4.42e-6
    return p


def density_unesco_eos80(s, t, p_dbar):
    """
    EOS-80 / UNESCO 近似：
    rho = rho(S, T, P)
    输出单位：kg/m^3
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    p_dbar = np.asarray(p_dbar, dtype=float)

    rho = np.full(np.broadcast(s, t, p_dbar).shape, np.nan, dtype=float)

    valid = (~np.isnan(s)) & (~np.isnan(t)) & (~np.isnan(p_dbar))
    if not np.any(valid):
        return rho

    sv = s[valid]
    tv = t[valid]
    pv_bar = p_dbar[valid] / 10.0

    rho_w = (
        999.842594
        + 6.793952e-2 * tv
        - 9.095290e-3 * tv**2
        + 1.001685e-4 * tv**3
        - 1.120083e-6 * tv**4
        + 6.536332e-9 * tv**5
    )

    A = (
        8.24493e-1
        - 4.0899e-3 * tv
        + 7.6438e-5 * tv**2
        - 8.2467e-7 * tv**3
        + 5.3875e-9 * tv**4
    )

    B = (
        -5.72466e-3
        + 1.0227e-4 * tv
        - 1.6546e-6 * tv**2
    )

    C = 4.8314e-4

    rho0 = rho_w + A * sv + B * (sv ** 1.5) + C * sv**2

    K_w = (
        19652.21
        + 148.4206 * tv
        - 2.327105 * tv**2
        + 1.360477e-2 * tv**3
        - 5.155288e-5 * tv**4
    )

    A_w = (
        54.6746
        - 0.603459 * tv
        + 1.09987e-2 * tv**2
        - 6.1670e-5 * tv**3
    )

    B_w = (
        7.944e-2
        + 1.6483e-2 * tv
        - 5.3009e-4 * tv**2
    )

    K0 = K_w + A_w * sv + B_w * (sv ** 1.5)

    Aw_p = (
        3.239908
        + 1.43713e-3 * tv
        + 1.16092e-4 * tv**2
        - 5.77905e-7 * tv**3
    )

    Bw_p = (
        2.2838e-3
        - 1.0981e-5 * tv
        - 1.6078e-6 * tv**2
    )

    Cw_p = 1.91075e-4

    Dw_p = (
        8.50935e-5
        - 6.12293e-6 * tv
        + 5.2787e-8 * tv**2
    )

    Ew_p = (
        -9.9348e-7
        + 2.0816e-8 * tv
        + 9.1697e-10 * tv**2
    )

    K = (
        K0
        + (Aw_p + Bw_p * sv + Cw_p * (sv ** 1.5)) * pv_bar
        + (Dw_p + Ew_p * sv) * pv_bar**2
    )

    rho_valid = rho0 / (1 - pv_bar / K)
    rho[valid] = rho_valid
    return rho


def compute_baseline_from_matrix(matrix, method="mean"):
    """
    对 shape=(n_profile, n_depth) 的矩阵按深度求基线。
    自动跳过全 NaN 列，避免 Mean of empty slice。
    """
    matrix = np.asarray(matrix, dtype=float)

    n_valid = np.sum(~np.isnan(matrix), axis=0)
    baseline_raw = np.full(matrix.shape[1], np.nan, dtype=float)

    valid_cols = n_valid > 0

    if method == "mean":
        if np.any(valid_cols):
            baseline_raw[valid_cols] = np.nanmean(matrix[:, valid_cols], axis=0)
    elif method == "median":
        if np.any(valid_cols):
            baseline_raw[valid_cols] = np.nanmedian(matrix[:, valid_cols], axis=0)
    else:
        raise ValueError(f"未知 baseline_method: {method}")

    return baseline_raw, n_valid


def get_top_features(df, has_kd_fast=False):
    """
    每条 profile 提取顶端一行，并计算顶端压强和密度。
    同时把顶端 slow/fast Kd 也保存下来。
    """
    top_df = (
        df.sort_values([profile_col, depth_col])
        .groupby(profile_col, as_index=False)
        .first()
    )

    keep_cols = [profile_col, depth_col, chl_col, temp_col, sal_col, kd_slow_col]
    if has_kd_fast:
        keep_cols.append(kd_fast_col)

    top_df = top_df[keep_cols].rename(columns={
        depth_col: "top_depth",
        chl_col: "top_chl",
        temp_col: "top_temperature",
        sal_col: "top_salinity",
        kd_slow_col: "top_kd_slow"
    })

    if has_kd_fast:
        top_df = top_df.rename(columns={kd_fast_col: "top_kd_fast"})
    else:
        top_df["top_kd_fast"] = np.nan

    top_df["top_pressure_dbar"] = depth_to_pressure_dbar(
        top_df["top_depth"].to_numpy(dtype=float),
        latitude_deg=latitude_deg
    )
    top_df["top_pressure_bar"] = top_df["top_pressure_dbar"] / 10.0

    top_df["top_density"] = density_unesco_eos80(
        top_df["top_salinity"].to_numpy(dtype=float),
        top_df["top_temperature"].to_numpy(dtype=float),
        top_df["top_pressure_dbar"].to_numpy(dtype=float)
    )
    top_df["top_sigma_t"] = top_df["top_density"] - 1000.0

    return top_df


def add_primary_alias_columns(df, primary_branch="slow"):
    """
    为了兼容旧脚本，生成 generic 别名列：
        kd_interp / Y_interp / B_y_raw / B_y / b_depthwise / ...
    默认映射到 slow。
    """
    out = df.copy()

    if primary_branch == "slow":
        out["kd_interp"] = out["kd_slow_interp"]
        out["Y_interp"] = out["Y_slow_interp"]
        out["B_y_raw"] = out["B_y_slow_raw"]
        out["B_y"] = out["B_y_slow"]
        out["b_depthwise"] = out["b_slow_depthwise"]
        out["b_depthwise_raw_baseline"] = out["b_slow_depthwise_raw_baseline"]
        out["is_valid_kd_interp"] = out["is_valid_kd_slow_interp"]
        out["n_valid_profiles_y"] = out["n_valid_profiles_y_slow"]

    elif primary_branch == "fast":
        out["kd_interp"] = out["kd_fast_interp"]
        out["Y_interp"] = out["Y_fast_interp"]
        out["B_y_raw"] = out["B_y_fast_raw"]
        out["B_y"] = out["B_y_fast"]
        out["b_depthwise"] = out["b_fast_depthwise"]
        out["b_depthwise_raw_baseline"] = out["b_fast_depthwise_raw_baseline"]
        out["is_valid_kd_interp"] = out["is_valid_kd_fast_interp"]
        out["n_valid_profiles_y"] = out["n_valid_profiles_y_fast"]

    else:
        raise ValueError(f"未知 primary_kd_branch: {primary_branch}")

    return out


# =========================================================
# 5. 读取数据
# =========================================================

df_raw = pd.read_csv(input_csv)

has_kd_fast = enable_kd_fast_if_exists and (kd_fast_col in df_raw.columns)

required_cols = [
    profile_col, depth_col,
    kd_slow_col, chl_col, temp_col, sal_col
]

missing_cols = [col for col in required_cols if col not in df_raw.columns]
if missing_cols:
    raise ValueError(f"输入文件缺少必要列: {missing_cols}")

use_cols = required_cols.copy()
if has_kd_fast:
    use_cols.append(kd_fast_col)

df = df_raw[use_cols].copy()

# 基本列不能缺
df = df.dropna(subset=[profile_col, depth_col]).copy()

# 把无效 Kd 设为 NaN
df.loc[df[kd_slow_col] <= 0, kd_slow_col] = np.nan
if has_kd_fast:
    df.loc[df[kd_fast_col] <= 0, kd_fast_col] = np.nan

# 至少 slow 或 fast 有一个有效
if has_kd_fast:
    df = df[df[kd_slow_col].notna() | df[kd_fast_col].notna()].copy()
else:
    df = df[df[kd_slow_col].notna()].copy()

if exclude_profile_ids:
    df = df[~df[profile_col].isin(exclude_profile_ids)].copy()

print("读取数据完成")
print("数据维度：", df.shape)
print("profile 数量：", df[profile_col].nunique())
print("深度范围：", df[depth_col].min(), "到", df[depth_col].max())
print("has_kd_fast =", has_kd_fast)


# =========================================================
# 6. 构造统一深度网格
# =========================================================

depth_grid = np.arange(depth_start, depth_end + 1e-9, depth_step)

print("\n统一深度网格：")
print("depth_start =", depth_start)
print("depth_end =", depth_end)
print("depth_step =", depth_step)
print("depth_grid 点数 =", len(depth_grid))


# =========================================================
# 7. 提取顶端特征
# =========================================================

top_df = get_top_features(df, has_kd_fast=has_kd_fast)

print("\n顶端特征表：")
print(top_df.head())


# =========================================================
# 8. 插值每条 profile 的各变量
# =========================================================

profile_ids = sorted(df[profile_col].unique().tolist())

interp_records = []

KD_SLOW_matrix = []
Y_SLOW_matrix = []

if has_kd_fast:
    KD_FAST_matrix = []
    Y_FAST_matrix = []

CHL_matrix = []
TEMP_matrix = []
SAL_matrix = []
DENSITY_matrix = []

valid_profile_ids = []

pressure_dbar_grid = depth_to_pressure_dbar(depth_grid, latitude_deg=latitude_deg)
pressure_bar_grid = pressure_dbar_grid / 10.0

for pid in profile_ids:
    profile_df = df[df[profile_col] == pid].copy().sort_values(depth_col)

    z = profile_df[depth_col].to_numpy(dtype=float)

    # slow
    kd_slow = profile_df[kd_slow_col].to_numpy(dtype=float)
    y_slow = transform_kd(kd_slow, method=kd_transform)
    kd_slow_interp = interp_or_nan(z, kd_slow, depth_grid, mode=out_of_range_mode)
    y_slow_interp = interp_or_nan(z, y_slow, depth_grid, mode=out_of_range_mode)

    # fast
    if has_kd_fast:
        kd_fast = profile_df[kd_fast_col].to_numpy(dtype=float)
        y_fast = transform_kd(kd_fast, method=kd_transform)
        kd_fast_interp = interp_or_nan(z, kd_fast, depth_grid, mode=out_of_range_mode)
        y_fast_interp = interp_or_nan(z, y_fast, depth_grid, mode=out_of_range_mode)
    else:
        kd_fast_interp = np.full(len(depth_grid), np.nan)
        y_fast_interp = np.full(len(depth_grid), np.nan)

    # 其它变量
    chl = profile_df[chl_col].to_numpy(dtype=float)
    temp = profile_df[temp_col].to_numpy(dtype=float)
    sal = profile_df[sal_col].to_numpy(dtype=float)

    chl_interp = interp_or_nan(z, chl, depth_grid, mode=out_of_range_mode)
    temp_interp = interp_or_nan(z, temp, depth_grid, mode=out_of_range_mode)
    sal_interp = interp_or_nan(z, sal, depth_grid, mode=out_of_range_mode)

    density_interp = density_unesco_eos80(
        sal_interp,
        temp_interp,
        pressure_dbar_grid
    )
    sigma_t_interp = density_interp - 1000.0

    # 至少 slow/fast 有一套有效
    if np.all(np.isnan(kd_slow_interp)) and np.all(np.isnan(kd_fast_interp)):
        print(f"跳过 profile {pid}：slow 和 fast 都无法在统一深度网格上形成有效插值")
        continue

    valid_profile_ids.append(pid)

    KD_SLOW_matrix.append(kd_slow_interp)
    Y_SLOW_matrix.append(y_slow_interp)

    if has_kd_fast:
        KD_FAST_matrix.append(kd_fast_interp)
        Y_FAST_matrix.append(y_fast_interp)

    CHL_matrix.append(chl_interp)
    TEMP_matrix.append(temp_interp)
    SAL_matrix.append(sal_interp)
    DENSITY_matrix.append(density_interp)

    top_row = top_df[top_df[profile_col] == pid].iloc[0]

    for j, depth in enumerate(depth_grid):
        interp_records.append({
            profile_col: pid,
            "depth": depth,

            # slow
            "kd_slow_interp": kd_slow_interp[j],
            "Y_slow_interp": y_slow_interp[j],

            # fast
            "kd_fast_interp": kd_fast_interp[j],
            "Y_fast_interp": y_fast_interp[j],

            # 全深度插值变量
            "chl_interp": chl_interp[j],
            "temp_interp": temp_interp[j],
            "sal_interp": sal_interp[j],

            # 深度影响后的压强与密度
            "pressure_dbar_interp": pressure_dbar_grid[j],
            "pressure_bar_interp": pressure_bar_grid[j],
            "density_interp": density_interp[j],
            "sigma_t_interp": sigma_t_interp[j],

            # 顶端值
            "top_depth": top_row["top_depth"],
            "top_chl": top_row["top_chl"],
            "top_temperature": top_row["top_temperature"],
            "top_salinity": top_row["top_salinity"],
            "top_pressure_dbar": top_row["top_pressure_dbar"],
            "top_pressure_bar": top_row["top_pressure_bar"],
            "top_density": top_row["top_density"],
            "top_sigma_t": top_row["top_sigma_t"],
            "top_kd_slow": top_row["top_kd_slow"],
            "top_kd_fast": top_row["top_kd_fast"],

            # 质量标记
            "is_valid_kd_slow_interp": not np.isnan(kd_slow_interp[j]),
            "is_valid_kd_fast_interp": not np.isnan(kd_fast_interp[j]),
            "is_valid_chl_interp": not np.isnan(chl_interp[j]),
            "is_valid_temp_interp": not np.isnan(temp_interp[j]),
            "is_valid_sal_interp": not np.isnan(sal_interp[j]),
            "is_valid_density_interp": not np.isnan(density_interp[j]),

            # 元信息
            "kd_transform": kd_transform
        })

KD_SLOW_matrix = np.asarray(KD_SLOW_matrix, dtype=float)
Y_SLOW_matrix = np.asarray(Y_SLOW_matrix, dtype=float)

if has_kd_fast:
    KD_FAST_matrix = np.asarray(KD_FAST_matrix, dtype=float)
    Y_FAST_matrix = np.asarray(Y_FAST_matrix, dtype=float)

CHL_matrix = np.asarray(CHL_matrix, dtype=float)
TEMP_matrix = np.asarray(TEMP_matrix, dtype=float)
SAL_matrix = np.asarray(SAL_matrix, dtype=float)
DENSITY_matrix = np.asarray(DENSITY_matrix, dtype=float)

interp_df = pd.DataFrame(interp_records)

print("\n插值完成")
print("有效 profile 数量：", len(valid_profile_ids))
print("Y_SLOW_matrix shape：", Y_SLOW_matrix.shape)
if has_kd_fast:
    print("Y_FAST_matrix shape：", Y_FAST_matrix.shape)


# =========================================================
# 9. 计算每个深度上的基线
# =========================================================

# slow
B_y_slow_raw, n_valid_y_slow = compute_baseline_from_matrix(Y_SLOW_matrix, method=baseline_method)
B_y_slow = smooth_1d(B_y_slow_raw, window=baseline_smooth_window)

# fast
if has_kd_fast:
    B_y_fast_raw, n_valid_y_fast = compute_baseline_from_matrix(Y_FAST_matrix, method=baseline_method)
    B_y_fast = smooth_1d(B_y_fast_raw, window=baseline_smooth_window)
else:
    B_y_fast_raw = np.full(len(depth_grid), np.nan)
    B_y_fast = np.full(len(depth_grid), np.nan)
    n_valid_y_fast = np.zeros(len(depth_grid), dtype=int)

# 其它变量
B_chl_raw, n_valid_chl = compute_baseline_from_matrix(CHL_matrix, method=baseline_method)
B_temp_raw, n_valid_temp = compute_baseline_from_matrix(TEMP_matrix, method=baseline_method)
B_sal_raw, n_valid_sal = compute_baseline_from_matrix(SAL_matrix, method=baseline_method)
B_density_raw, n_valid_density = compute_baseline_from_matrix(DENSITY_matrix, method=baseline_method)

B_chl = smooth_1d(B_chl_raw, window=baseline_smooth_window)
B_temp = smooth_1d(B_temp_raw, window=baseline_smooth_window)
B_sal = smooth_1d(B_sal_raw, window=baseline_smooth_window)
B_density = smooth_1d(B_density_raw, window=baseline_smooth_window)

baseline_df = pd.DataFrame({
    "depth": depth_grid,

    "B_y_slow_raw": B_y_slow_raw,
    "B_y_slow": B_y_slow,

    "B_y_fast_raw": B_y_fast_raw,
    "B_y_fast": B_y_fast,

    "B_chl_raw": B_chl_raw,
    "B_chl": B_chl,

    "B_temp_raw": B_temp_raw,
    "B_temp": B_temp,

    "B_sal_raw": B_sal_raw,
    "B_sal": B_sal,

    "B_density_raw": B_density_raw,
    "B_density": B_density,

    "n_valid_profiles_y_slow": n_valid_y_slow,
    "n_valid_profiles_y_fast": n_valid_y_fast,
    "n_valid_profiles_chl": n_valid_chl,
    "n_valid_profiles_temp": n_valid_temp,
    "n_valid_profiles_sal": n_valid_sal,
    "n_valid_profiles_density": n_valid_density,

    # generic 别名
    "baseline_method": baseline_method,
    "baseline_smooth_window": baseline_smooth_window,
    "baseline_kd_transform": kd_transform
})

# generic baseline 别名
if primary_kd_branch == "slow":
    baseline_df["B_y_raw"] = baseline_df["B_y_slow_raw"]
    baseline_df["B_y"] = baseline_df["B_y_slow"]
    baseline_df["n_valid_profiles_y"] = baseline_df["n_valid_profiles_y_slow"]
elif primary_kd_branch == "fast":
    baseline_df["B_y_raw"] = baseline_df["B_y_fast_raw"]
    baseline_df["B_y"] = baseline_df["B_y_fast"]
    baseline_df["n_valid_profiles_y"] = baseline_df["n_valid_profiles_y_fast"]
else:
    raise ValueError(f"未知 primary_kd_branch: {primary_kd_branch}")

if drop_depths_with_zero_valid_profiles:
    baseline_df = baseline_df[baseline_df["n_valid_profiles_y"] > 0].copy()
    baseline_df = baseline_df.reset_index(drop=True)

baseline_df.to_csv(baseline_output_csv, index=False, encoding="utf-8-sig")

print("\n深度基线表已保存：")
print(baseline_output_csv)


# =========================================================
# 10. 合并基线并计算偏移
# =========================================================

interp_df = interp_df.merge(
    baseline_df,
    on="depth",
    how="left"
)

# slow 偏移
interp_df["b_slow_depthwise"] = interp_df["Y_slow_interp"] - interp_df["B_y_slow"]
interp_df["b_slow_depthwise_raw_baseline"] = interp_df["Y_slow_interp"] - interp_df["B_y_slow_raw"]

# fast 偏移
interp_df["b_fast_depthwise"] = interp_df["Y_fast_interp"] - interp_df["B_y_fast"]
interp_df["b_fast_depthwise_raw_baseline"] = interp_df["Y_fast_interp"] - interp_df["B_y_fast_raw"]

# 定义
interp_df["b_definition_slow"] = "b_slow_i(z)=Y_slow_i(z)-B_y_slow(z)"
interp_df["b_definition_fast"] = "b_fast_i(z)=Y_fast_i(z)-B_y_fast(z)"

# anomaly
interp_df["chl_anomaly"] = interp_df["chl_interp"] - interp_df["B_chl"]
interp_df["temp_anomaly"] = interp_df["temp_interp"] - interp_df["B_temp"]
interp_df["sal_anomaly"] = interp_df["sal_interp"] - interp_df["B_sal"]
interp_df["density_anomaly"] = interp_df["density_interp"] - interp_df["B_density"]

# generic 别名默认映射到 slow
interp_df = add_primary_alias_columns(interp_df, primary_branch=primary_kd_branch)


# =========================================================
# 11. 主表输出：尽量保留所有信息
# =========================================================

master_cols = [
    profile_col,
    "depth",

    # 顶端特征
    "top_depth",
    "top_chl",
    "top_temperature",
    "top_salinity",
    "top_pressure_dbar",
    "top_pressure_bar",
    "top_density",
    "top_sigma_t",
    "top_kd_slow",
    "top_kd_fast",

    # generic 别名（兼容旧脚本）
    "kd_interp",
    "Y_interp",
    "B_y_raw",
    "B_y",
    "b_depthwise",
    "b_depthwise_raw_baseline",
    "is_valid_kd_interp",
    "n_valid_profiles_y",

    # slow 分支
    "kd_slow_interp",
    "Y_slow_interp",
    "B_y_slow_raw",
    "B_y_slow",
    "b_slow_depthwise",
    "b_slow_depthwise_raw_baseline",
    "is_valid_kd_slow_interp",
    "n_valid_profiles_y_slow",
    "b_definition_slow",

    # fast 分支
    "kd_fast_interp",
    "Y_fast_interp",
    "B_y_fast_raw",
    "B_y_fast",
    "b_fast_depthwise",
    "b_fast_depthwise_raw_baseline",
    "is_valid_kd_fast_interp",
    "n_valid_profiles_y_fast",
    "b_definition_fast",

    # 全深度插值变量
    "chl_interp",
    "temp_interp",
    "sal_interp",
    "pressure_dbar_interp",
    "pressure_bar_interp",
    "density_interp",
    "sigma_t_interp",

    # 各变量基线
    "B_chl_raw",
    "B_chl",
    "B_temp_raw",
    "B_temp",
    "B_sal_raw",
    "B_sal",
    "B_density_raw",
    "B_density",

    # anomaly
    "chl_anomaly",
    "temp_anomaly",
    "sal_anomaly",
    "density_anomaly",

    # 有效样本数
    "n_valid_profiles_chl",
    "n_valid_profiles_temp",
    "n_valid_profiles_sal",
    "n_valid_profiles_density",

    # 质量控制
    "is_valid_chl_interp",
    "is_valid_temp_interp",
    "is_valid_sal_interp",
    "is_valid_density_interp",

    # 元信息
    "baseline_method",
    "baseline_smooth_window",
    "baseline_kd_transform",
    "kd_transform"
]

missing_master_cols = [c for c in master_cols if c not in interp_df.columns]
if missing_master_cols:
    raise ValueError(f"主表缺少这些列: {missing_master_cols}")

master_df = interp_df[master_cols].copy()
master_df.to_csv(master_output_csv, index=False, encoding="utf-8-sig")

print("\n主表已保存：")
print(master_output_csv)


# =========================================================
# 12. GAM 训练用表
# 说明：
# - slow 和 fast 都保留在一个 csv 中
# - generic 别名默认仍映射到 slow
# =========================================================

gam_df = master_df.copy()

if drop_nan_for_training:
    # 这里只按顶端叶绿素与至少一套 slow/fast 有效来删行
    key_mask = gam_df["top_chl"].notna()

    slow_valid_mask = (
        gam_df["kd_slow_interp"].notna()
        & gam_df["Y_slow_interp"].notna()
        & gam_df["B_y_slow"].notna()
        & gam_df["b_slow_depthwise"].notna()
    )

    fast_valid_mask = (
        gam_df["kd_fast_interp"].notna()
        & gam_df["Y_fast_interp"].notna()
        & gam_df["B_y_fast"].notna()
        & gam_df["b_fast_depthwise"].notna()
    )

    gam_df = gam_df[key_mask & (slow_valid_mask | fast_valid_mask)].copy()

gam_df.to_csv(train_output_csv, index=False, encoding="utf-8-sig")

print("\nGAM 训练用表已保存（slow + fast 在同一个 csv 中）：")
print(train_output_csv)

print("\n训练表维度：", gam_df.shape)
print("\n训练表前10行：")
print(gam_df.head(10))


# =========================================================
# 13. 简单检查
# =========================================================

print("\n主表列名：")
print(master_df.columns.tolist())

print("\n每个深度有效 profile 数量（slow / fast）：")
print(
    baseline_df[
        ["depth", "n_valid_profiles_y_slow", "n_valid_profiles_y_fast"]
    ].head(20)
)

print("\nslow b_depthwise 描述统计：")
print(gam_df["b_slow_depthwise"].describe())

if has_kd_fast:
    print("\nfast b_depthwise 描述统计：")
    print(gam_df["b_fast_depthwise"].describe())

print("\nchl_interp 描述统计：")
print(gam_df["chl_interp"].describe())

print("\ndensity_interp 描述统计：")
print(gam_df["density_interp"].describe())

print("\n全部完成。")