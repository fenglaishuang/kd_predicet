# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib

# 后台保存图片，不弹出窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from pathlib import Path


# ============================================================
# 1. 文件路径与输出路径
# ============================================================

# KD 类型：
# "slow"：绘制 Kdslow
# "fast"：绘制 Kdfast
KD_TYPE = "slow"

# 重构结果文件
RECON_CSV = r"D:\work\reconstruct_slow_3layers_transition\profile_plot_points_slow.csv"

# mat 文件：只读取经纬度，不读取现场叶绿素
MAT_INFO_FILE = r"D:\work\经纬度和现场叶绿素.mat"

# 图片输出目录，可自行指定
OUTPUT_DIR = r"D:\work\reconstruct_all_profiles_slow_ac_compatible\figures"

TRUE_COL_OVERRIDE = "kd_true"
PRED_COL_OVERRIDE = "kd_reconstructed"
BASELINE_COL_OVERRIDE = "baseline_y"

OUT_DIR = Path(OUTPUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 绘图模式设置
# ============================================================

# 可选：
# "single"：绘制单张剖面图
# "grid"  ：绘制组合图
PLOT_MODE = "grid"

# 单图模式使用
PROFILE_ID = 1

# 组合图模式使用
GRID_PROFILE_IDS = [3, 7, 8, 16, 24, 25]
GRID_NROWS = 2
GRID_NCOLS = 3


# ============================================================
# 2B. 深度轴范围控制
# ============================================================

# "individual"：每个子图按自身深度范围显示，适合不同剖面深度不一致
# "global"    ：所有子图使用统一深度范围，便于横向比较
# "manual"    ：所有子图使用手动深度范围
GRID_DEPTH_MODE = "individual"

# 单图深度轴模式：
# "individual"：按该剖面自身深度范围
# "manual"    ：使用手动深度范围
SINGLE_DEPTH_MODE = "individual"

# 仅在 *_DEPTH_MODE = "manual" 时生效
# 写成 (浅层深度, 深层深度)，例如 (0, 80)
MANUAL_DEPTH_RANGE = (0, 80)

# 深度轴留白比例
DEPTH_PAD_RATIO = 0.03


# ============================================================
# 3. 字体与字号参数区
# ============================================================

FONT_FAMILY = "Times New Roman"
MATH_FONTSET = "custom"

GLOBAL_FONT_SIZE = 11

SINGLE_AXIS_LABEL_FONTSIZE = 16
GRID_AXIS_LABEL_FONTSIZE = 15
AXIS_LABEL_FONTWEIGHT = "bold"

XTICK_FONTSIZE = 12
YTICK_FONTSIZE = 12
TICK_FONTWEIGHT = "bold"

MAJOR_TICK_WIDTH = 1.2
MINOR_TICK_WIDTH = 1.0

SINGLE_TITLE_FONTSIZE = 14
GRID_TITLE_FONTSIZE = 10.5
SUPTITLE_FONTSIZE = 16

SINGLE_TITLE_FONTWEIGHT = "bold"
GRID_TITLE_FONTWEIGHT = "bold"
SUPTITLE_FONTWEIGHT = "bold"

SINGLE_LEGEND_FONTSIZE = 14
GRID_LEGEND_FONTSIZE = 14
LEGEND_FONTWEIGHT = "normal"

NO_DATA_FONTSIZE = 10
NO_DATA_FONTWEIGHT = "normal"

FIGURE_DPI = 300
SAVEFIG_DPI = 600


# ============================================================
# 4. 可调绘图参数
# ============================================================

SHOW_BASELINE = True
USE_LOG_X = False

LINEWIDTH_TRUE = 2.0
LINEWIDTH_PRED = 2.0
LINEWIDTH_BASELINE = 1.7

ALPHA_TRUE_LINE = 0.90
ALPHA_PRED_LINE = 0.85
ALPHA_BASELINE = 1.00

MARKERSIZE_TRUE = 3.0
MARKERSIZE_PRED = 3.0

FILL_ALPHA = 0.08

SINGLE_LEGEND_LOC = "center right"
SINGLE_LEGEND_BBOX = (0.98, 0.56)

# 默认图例位置：如果没有单独指定某个 profile，则使用这里
GRID_LEGEND_LOC_TOP = "lower right"
GRID_LEGEND_BBOX_TOP = (0.98, 0.02)

GRID_LEGEND_LOC_BOTTOM = "lower left"
GRID_LEGEND_BBOX_BOTTOM = (0.02, 0.02)

# 组合图尺寸：第二个数越大，纵向越长
GRID_FIGSIZE = (18, 14)
SINGLE_FIGSIZE = (7.0, 11.0)


# ============================================================
# 4B. 每个子图单独设置图例位置
# ============================================================

USE_CUSTOM_LEGEND_POSITION = True

# loc 是图例锚点，bbox 是子图内部相对坐标：
# 右下角：loc="lower right", bbox=(0.98, 0.02)
# 左下角：loc="lower left",  bbox=(0.02, 0.02)
# 右上角：loc="upper right", bbox=(0.98, 0.98)
# 左上角：loc="upper left",  bbox=(0.02, 0.98)
CUSTOM_LEGEND_POSITIONS = {
    # 左下角
    3:  {"loc": "lower left",  "bbox": (0.02, 0.02)},

    # 右下角
    7:  {"loc": "center right", "bbox": (0.98, 0.50)},

    # 右侧中部
    8:  {"loc": "center right", "bbox": (0.98, 0.50)},

    # 右下角
    16: {"loc": "lower left", "bbox": (0.02, 0.02)},

    # 右下角
    24: {"loc": "lower right", "bbox": (0.98, 0.02)},

    #
    25: {"loc": "lower right",  "bbox": (0.98, 0.02)},
}


# ============================================================
# 5. RGB 三维坐标颜色函数
# ============================================================

def rgb255(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


# ============================================================
# 6. 颜色设置
# ============================================================

# 原始 KD：蓝色
COLOR_TRUE_LINE = rgb255(13, 117, 193)

# 预测 KD：红色
COLOR_PRED_LINE = rgb255(255, 80, 84)

# 深度基线：绿色
COLOR_BASELINE = rgb255(103, 194, 165)

# 阴影颜色
COLOR_FILL = COLOR_PRED_LINE

COLOR_GRID = rgb255(207, 207, 207)
COLOR_AX_BG = rgb255(255, 255, 255)
COLOR_SPINE = rgb255(64, 64, 64)
COLOR_LEGEND_EDGE = rgb255(153, 153, 153)


# ============================================================
# 7. 全局绘图风格
# ============================================================

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": MATH_FONTSET,
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": FONT_FAMILY + ":italic",
    "mathtext.bf": FONT_FAMILY + ":bold",
    "axes.unicode_minus": False,
    "font.size": GLOBAL_FONT_SIZE,
    "axes.labelsize": GRID_AXIS_LABEL_FONTSIZE,
    "axes.titlesize": GRID_TITLE_FONTSIZE,
    "xtick.labelsize": XTICK_FONTSIZE,
    "ytick.labelsize": YTICK_FONTSIZE,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": SAVEFIG_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 8. 基础工具函数
# ============================================================

def first_valid(series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan


def to_1d_array(x):
    arr = np.asarray(x)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        arr = np.array([arr.item()])
    elif arr.ndim > 1:
        arr = arr.reshape(-1)

    return arr


def find_key_ignore_case(data_dict, candidates):
    lower_map = {str(k).lower(): k for k in data_dict.keys()}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def find_first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_lon_lat(lon, lat):
    if np.isfinite(lon) and np.isfinite(lat):
        lon_text = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
        lat_text = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
        return f"{lon_text}, {lat_text}"

    return "Lon/Lat: NA"


def get_kd_tex(kd_type):
    kd_type = kd_type.lower()

    if kd_type == "slow":
        return r"K_{d,slow}"
    elif kd_type == "fast":
        return r"K_{d,fast}"
    else:
        return r"K_d"


def get_kd_name_tag(kd_type):
    kd_type = kd_type.lower()

    if kd_type == "slow":
        return "kdslow"
    elif kd_type == "fast":
        return "kdfast"
    else:
        return "kd"


def apply_tick_font(ax):
    ax.tick_params(axis="x", which="major", labelsize=XTICK_FONTSIZE, width=MAJOR_TICK_WIDTH)
    ax.tick_params(axis="y", which="major", labelsize=YTICK_FONTSIZE, width=MAJOR_TICK_WIDTH)
    ax.tick_params(axis="both", which="minor", width=MINOR_TICK_WIDTH)

    for tick in ax.get_xticklabels():
        tick.set_fontname(FONT_FAMILY)
        tick.set_fontweight(TICK_FONTWEIGHT)
        tick.set_fontsize(XTICK_FONTSIZE)

    for tick in ax.get_yticklabels():
        tick.set_fontname(FONT_FAMILY)
        tick.set_fontweight(TICK_FONTWEIGHT)
        tick.set_fontsize(YTICK_FONTSIZE)


def apply_legend_font(legend):
    if legend is None:
        return

    for text in legend.get_texts():
        text.set_fontname(FONT_FAMILY)
        text.set_fontweight(LEGEND_FONTWEIGHT)


def get_grid_legend_position(idx, profile_id):
    if USE_CUSTOM_LEGEND_POSITION:
        cfg = CUSTOM_LEGEND_POSITIONS.get(int(profile_id), None)
        if cfg is not None:
            return cfg["loc"], cfg["bbox"]

    if idx < GRID_NCOLS:
        return GRID_LEGEND_LOC_TOP, GRID_LEGEND_BBOX_TOP

    return GRID_LEGEND_LOC_BOTTOM, GRID_LEGEND_BBOX_BOTTOM


# ============================================================
# 9. 根据 KD_TYPE 自动选择列
# ============================================================

def resolve_kd_columns(df, kd_type):
    kd_type = kd_type.lower()

    if kd_type not in ["slow", "fast"]:
        raise ValueError("KD_TYPE 只能是 'slow' 或 'fast'。")

    if kd_type == "slow":
        true_candidates = [
            "kd_true",
            "kd_slow_true",
            "kd_slow_interp",
            "Y_slow_interp",
            "Y_interp",
            "kd_interp",
        ]

        pred_candidates = [
            "kd_reconstructed",
            "kd_slow_reconstructed",
            "kd_reconstructed_slow",
            "reconstructed_kd_slow",
            "pred_kd_slow",
            "kd_slow_pred",
        ]

        baseline_candidates = [
            "baseline_y",
            "B_y_slow",
            "B_y_slow_raw",
            "B_y",
        ]

    else:
        true_candidates = [
            "kd_fast_true",
            "kd_fast_interp",
            "Y_fast_interp",
            "kd_fast",
            "Y_fast",
        ]

        pred_candidates = [
            "kd_fast_reconstructed",
            "kd_reconstructed_fast",
            "reconstructed_kd_fast",
            "pred_kd_fast",
            "kd_fast_pred",
            "kd_reconstructed",
        ]

        baseline_candidates = [
            "baseline_y_fast",
            "B_y_fast",
            "B_y_fast_raw",
            "baseline_y",
        ]

    true_col = TRUE_COL_OVERRIDE or find_first_existing_col(df, true_candidates)
    pred_col = PRED_COL_OVERRIDE or find_first_existing_col(df, pred_candidates)
    baseline_col = BASELINE_COL_OVERRIDE or find_first_existing_col(df, baseline_candidates)

    if true_col is None:
        raise ValueError(
            f"未找到 {kd_type} 的真实值列。\n"
            f"请检查文件列名，或设置 TRUE_COL_OVERRIDE。\n"
            f"当前文件列名：{df.columns.tolist()}"
        )

    if pred_col is None:
        raise ValueError(
            f"未找到 {kd_type} 的重构/预测值列。\n"
            f"请检查文件列名，或设置 PRED_COL_OVERRIDE。\n"
            f"当前文件列名：{df.columns.tolist()}"
        )

    if baseline_col is None and SHOW_BASELINE:
        print(f"提示：未找到 {kd_type} 的基线列，将不绘制深度基线。")

    print("当前绘图列：")
    print(f"KD_TYPE       = {kd_type}")
    print(f"TRUE_COL      = {true_col}")
    print(f"PRED_COL      = {pred_col}")
    print(f"BASELINE_COL  = {baseline_col}")

    return true_col, pred_col, baseline_col


# ============================================================
# 10. 读取 mat 中的经纬度信息
# ============================================================

def load_mat_file(mat_path):
    mat_path = Path(mat_path)

    if not mat_path.exists():
        print(f"提示：未找到 mat 文件：{mat_path}")
        return None

    try:
        from scipy.io import loadmat

        mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        mat = {
            k: v for k, v in mat.items()
            if not k.startswith("__")
        }

        return mat

    except NotImplementedError:
        pass

    except Exception as e:
        print(f"提示：scipy 读取 mat 文件失败，尝试 h5py。原因：{e}")

    try:
        import h5py

        out = {}

        with h5py.File(str(mat_path), "r") as f:
            for key in f.keys():
                try:
                    arr = np.array(f[key])
                    arr = np.squeeze(arr)
                    out[key] = arr
                except Exception:
                    pass

        return out

    except Exception as e:
        print(f"提示：h5py 读取 mat 文件失败：{e}")
        return None


def extract_lon_lat_from_combined_array(mat, combined_key, default_order="lonlat"):
    arr = np.asarray(mat[combined_key])
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None, None

    if arr.shape[1] == 2:
        first = arr[:, 0]
        second = arr[:, 1]
    elif arr.shape[0] == 2:
        first = arr[0, :]
        second = arr[1, :]
    else:
        return None, None

    key_lower = str(combined_key).lower()

    if "latlon" in key_lower or "lat_lon" in key_lower or default_order == "latlon":
        lat_arr = to_1d_array(first).astype(float)
        lon_arr = to_1d_array(second).astype(float)
    else:
        lon_arr = to_1d_array(first).astype(float)
        lat_arr = to_1d_array(second).astype(float)

    return lon_arr, lat_arr


def build_profile_info_from_mat(mat_path, df_all):
    mat = load_mat_file(mat_path)

    if mat is None:
        return {}

    print("mat 文件变量名：")
    print(list(mat.keys()))

    profile_id_candidates = [
        "profile_id", "profileid", "profile", "id", "ID",
        "station_id", "stationid",
        "cast_id", "castid",
    ]

    lon_candidates = [
        "lon", "Lon", "LON",
        "longitude", "Longitude", "LONGITUDE",
        "lng", "Lng", "LNG",
        "经度",
        "jingdu", "jd",
        "x", "X",
    ]

    lat_candidates = [
        "lat", "Lat", "LAT",
        "latitude", "Latitude", "LATITUDE",
        "纬度",
        "weidu", "wd",
        "y", "Y",
    ]

    combined_lonlat_candidates = [
        "lonlat", "lon_lat", "longitude_latitude",
        "latlon", "lat_lon", "latitude_longitude",
        "经纬度", "jingweidu",
        "coords", "coordinates",
        "location", "locations",
        "pos", "position",
    ]

    profile_key = find_key_ignore_case(mat, profile_id_candidates)
    lon_key = find_key_ignore_case(mat, lon_candidates)
    lat_key = find_key_ignore_case(mat, lat_candidates)

    lon_arr = None
    lat_arr = None
    used_source = None

    if lon_key is not None and lat_key is not None:
        lon_arr = to_1d_array(mat[lon_key]).astype(float)
        lat_arr = to_1d_array(mat[lat_key]).astype(float)
        used_source = f"lon={lon_key}, lat={lat_key}"

    if lon_arr is None or lat_arr is None:
        combined_key = find_key_ignore_case(mat, combined_lonlat_candidates)

        if combined_key is not None:
            lon_arr, lat_arr = extract_lon_lat_from_combined_array(
                mat=mat,
                combined_key=combined_key,
                default_order="lonlat"
            )

            if lon_arr is not None and lat_arr is not None:
                used_source = f"combined_lonlat={combined_key}"

    if lon_arr is None or lat_arr is None:
        print("提示：mat 文件中未找到经纬度变量。")
        return {}

    n = min(len(lon_arr), len(lat_arr))
    profile_info = {}

    if profile_key is not None:
        profile_arr = to_1d_array(mat[profile_key]).astype(int)
        n = min(n, len(profile_arr))

        for i in range(n):
            pid = int(profile_arr[i])
            profile_info[pid] = {
                "lon": float(lon_arr[i]),
                "lat": float(lat_arr[i]),
            }

        print(f"已从 mat 文件按 profile_id 匹配经纬度，共匹配 {len(profile_info)} 个剖面。")
        print(f"使用变量：profile_id={profile_key}, {used_source}")

        return profile_info

    unique_profile_ids = sorted(
        df_all["profile_id"].dropna().unique().astype(int).tolist()
    )

    n = min(n, len(unique_profile_ids))

    for i in range(n):
        pid = int(unique_profile_ids[i])
        profile_info[pid] = {
            "lon": float(lon_arr[i]),
            "lat": float(lat_arr[i]),
        }

    print(f"mat 文件中未发现 profile_id，已按 profile_id 排序顺序匹配经纬度，共匹配 {len(profile_info)} 个剖面。")
    print(f"使用变量：{used_source}")

    return profile_info


def get_profile_lon_lat_from_info(profile_info, profile_id):
    info = profile_info.get(int(profile_id), None)

    if info is None:
        return np.nan, np.nan

    return info.get("lon", np.nan), info.get("lat", np.nan)


# ============================================================
# 11. 剖面数据与坐标轴工具函数
# ============================================================

def get_profile_df(df_all, profile_id, true_col, pred_col):
    profile_df = df_all[df_all["profile_id"].astype(int) == int(profile_id)].copy()

    profile_df = profile_df[
        profile_df[true_col].notna() &
        profile_df[pred_col].notna() &
        profile_df["depth"].notna()
    ].copy()

    if profile_df.empty:
        return None

    return profile_df.sort_values("depth")


def style_axis(ax):
    ax.set_facecolor(COLOR_AX_BG)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.40,
        color=COLOR_GRID
    )

    if USE_LOG_X:
        ax.set_xscale("log")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(COLOR_SPINE)

    apply_tick_font(ax)


def calc_depth_ylim(depth_values, pad_ratio=0.03):
    depth_values = np.asarray(depth_values, dtype=float)
    depth_values = depth_values[np.isfinite(depth_values)]

    if len(depth_values) == 0:
        return None

    dmin = float(np.nanmin(depth_values))
    dmax = float(np.nanmax(depth_values))

    pad = pad_ratio * (dmax - dmin) if dmax > dmin else 0.5

    return dmax + pad, dmin - pad


def set_depth_ylim(ax, depth_values=None, manual_range=None, pad_ratio=0.03):
    if manual_range is not None:
        zmin, zmax = manual_range
        zmin = float(zmin)
        zmax = float(zmax)

        if zmin > zmax:
            zmin, zmax = zmax, zmin

        pad = pad_ratio * (zmax - zmin) if zmax > zmin else 0.5
        ax.set_ylim(zmax + pad, zmin - pad)
        return True

    ylim = calc_depth_ylim(depth_values, pad_ratio=pad_ratio)

    if ylim is None:
        return False

    ax.set_ylim(*ylim)
    return True


def set_x_lim(ax, kd_true, kd_pred, baseline=None, pad_ratio=0.10):
    all_x = [kd_true, kd_pred]

    if baseline is not None:
        all_x.append(baseline)

    x_all = np.concatenate(all_x)
    x_all = x_all[np.isfinite(x_all)]

    if len(x_all) == 0:
        return

    x_min = np.nanmin(x_all)
    x_max = np.nanmax(x_all)

    pad = pad_ratio * (x_max - x_min) if x_max > x_min else 0.02
    ax.set_xlim(max(0, x_min - pad), x_max + pad)


def make_profile_legend(
    ax,
    line_true,
    line_pred,
    baseline_handle,
    top_chl,
    zenith_deg,
    surface_kd,
    kd_tex,
    legend_loc,
    legend_bbox,
    legend_fontsize
):
    legend_handles = [line_true, line_pred]
    legend_labels = [
        rf"In-situ ${kd_tex}$",
        rf"Re. ${kd_tex}$",
    ]

    if baseline_handle is not None:
        legend_handles.append(baseline_handle)
        legend_labels.append("Depthwise baseline")

    meta_handles = [
        Line2D(
            [], [],
            linestyle="None",
            color="none",
            label=f"Surface Chl = {top_chl:.4f}" if np.isfinite(top_chl) else "Surface Chl = NA"
        ),
        Line2D(
            [], [],
            linestyle="None",
            color="none",
            label=f"Zenith = {zenith_deg:.2f}°" if np.isfinite(zenith_deg) else "Zenith = NA"
        ),
        Line2D(
            [], [],
            linestyle="None",
            color="none",
            label=f"Surface $K_d$ = {surface_kd:.4f}" if np.isfinite(surface_kd) else "Surface $K_d$ = NA"
        )
    ]

    legend_handles.extend(meta_handles)
    legend_labels.extend([h.get_label() for h in meta_handles])

    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        frameon=True,
        framealpha=0.90,
        edgecolor=COLOR_LEGEND_EDGE,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.35,
        fontsize=legend_fontsize
    )

    apply_legend_font(legend)


# ============================================================
# 12. 单个剖面绘图函数
# ============================================================

def plot_profile_on_ax(
    ax,
    profile_df,
    profile_id,
    profile_info,
    true_col,
    pred_col,
    baseline_col,
    kd_type,
    show_legend=True,
    legend_loc="center right",
    legend_bbox=(0.98, 0.56),
    legend_fontsize=8.0
):
    if profile_df is None or profile_df.empty:
        ax.text(
            0.5,
            0.5,
            f"Profile {profile_id}\nNo valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=NO_DATA_FONTSIZE,
            fontname=FONT_FAMILY,
            fontweight=NO_DATA_FONTWEIGHT
        )
        ax.set_axis_off()
        return None

    kd_tex = get_kd_tex(kd_type)

    depth = profile_df["depth"].to_numpy(dtype=float)
    kd_true = profile_df[true_col].to_numpy(dtype=float)
    kd_pred = profile_df[pred_col].to_numpy(dtype=float)

    baseline = None
    if SHOW_BASELINE and baseline_col is not None and baseline_col in profile_df.columns:
        baseline = profile_df[baseline_col].to_numpy(dtype=float)

    top_chl = first_valid(profile_df["top_chl"]) if "top_chl" in profile_df.columns else np.nan
    zenith_deg = first_valid(profile_df["zenith_deg"]) if "zenith_deg" in profile_df.columns else np.nan
    surface_kd = first_valid(profile_df["surface_kd"]) if "surface_kd" in profile_df.columns else np.nan

    ax.fill_betweenx(
        depth,
        kd_true,
        kd_pred,
        color=COLOR_FILL,
        alpha=FILL_ALPHA,
        zorder=1
    )

    line_true, = ax.plot(
        kd_true,
        depth,
        color=COLOR_TRUE_LINE,
        linewidth=LINEWIDTH_TRUE,
        alpha=ALPHA_TRUE_LINE,
        marker="o",
        markersize=MARKERSIZE_TRUE,
        markerfacecolor=COLOR_TRUE_LINE,
        markeredgecolor=COLOR_TRUE_LINE,
        label=rf"In-situ ${kd_tex}$",
        zorder=4
    )

    line_pred, = ax.plot(
        kd_pred,
        depth,
        color=COLOR_PRED_LINE,
        linewidth=LINEWIDTH_PRED,
        alpha=ALPHA_PRED_LINE,
        marker="s",
        markersize=MARKERSIZE_PRED,
        markerfacecolor=COLOR_PRED_LINE,
        markeredgecolor=COLOR_PRED_LINE,
        label=rf"Re. ${kd_tex}$",
        zorder=5
    )

    baseline_handle = None

    if baseline is not None:
        baseline_handle, = ax.plot(
            baseline,
            depth,
            color=COLOR_BASELINE,
            linestyle="--",
            linewidth=LINEWIDTH_BASELINE,
            alpha=ALPHA_BASELINE,
            label="Depthwise baseline",
            zorder=3
        )

    set_x_lim(ax, kd_true, kd_pred, baseline)
    style_axis(ax)

    lon, lat = get_profile_lon_lat_from_info(profile_info, profile_id)
    lon_lat_text = format_lon_lat(lon, lat)

    ax.set_title(
        f"Profile {profile_id}\n{lon_lat_text}",
        fontsize=GRID_TITLE_FONTSIZE,
        fontweight=GRID_TITLE_FONTWEIGHT,
        fontname=FONT_FAMILY
    )

    if show_legend:
        make_profile_legend(
            ax=ax,
            line_true=line_true,
            line_pred=line_pred,
            baseline_handle=baseline_handle,
            top_chl=top_chl,
            zenith_deg=zenith_deg,
            surface_kd=surface_kd,
            kd_tex=kd_tex,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            legend_fontsize=legend_fontsize
        )

    return {
        "line_true": line_true,
        "line_pred": line_pred,
        "baseline_handle": baseline_handle,
        "depth": depth,
    }


# ============================================================
# 13. 单图绘制函数
# ============================================================

def draw_single_profile(
    df_all,
    profile_id,
    profile_info,
    true_col,
    pred_col,
    baseline_col,
    kd_type
):
    profile_df = get_profile_df(
        df_all=df_all,
        profile_id=profile_id,
        true_col=true_col,
        pred_col=pred_col
    )

    if profile_df is None:
        raise ValueError(f"未找到 profile_id = {profile_id} 的有效数据。")

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)

    result = plot_profile_on_ax(
        ax=ax,
        profile_df=profile_df,
        profile_id=profile_id,
        profile_info=profile_info,
        true_col=true_col,
        pred_col=pred_col,
        baseline_col=baseline_col,
        kd_type=kd_type,
        show_legend=True,
        legend_loc=SINGLE_LEGEND_LOC,
        legend_bbox=SINGLE_LEGEND_BBOX,
        legend_fontsize=SINGLE_LEGEND_FONTSIZE
    )

    if SINGLE_DEPTH_MODE == "manual":
        set_depth_ylim(
            ax,
            manual_range=MANUAL_DEPTH_RANGE,
            pad_ratio=DEPTH_PAD_RATIO
        )
    else:
        set_depth_ylim(
            ax,
            depth_values=result["depth"],
            pad_ratio=DEPTH_PAD_RATIO
        )

    kd_tex = get_kd_tex(kd_type)
    lon, lat = get_profile_lon_lat_from_info(profile_info, profile_id)

    ax.set_xlabel(
        rf"${kd_tex}$ (m$^{{-1}}$)",
        fontsize=SINGLE_AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
        fontname=FONT_FAMILY
    )
    ax.set_ylabel(
        "Depth (m)",
        fontsize=SINGLE_AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
        fontname=FONT_FAMILY
    )

    ax.set_title(
        f"Profile {profile_id}: In-situ vs. reconstructed ${kd_tex}$\n"
        f"{format_lon_lat(lon, lat)}",
        fontsize=SINGLE_TITLE_FONTSIZE,
        fontweight=SINGLE_TITLE_FONTWEIGHT,
        fontname=FONT_FAMILY
    )

    kd_name_tag = get_kd_name_tag(kd_type)
    depth_mode_tag = SINGLE_DEPTH_MODE

    out_png = OUT_DIR / f"profile_{profile_id}_{kd_name_tag}_compare_{depth_mode_tag}_depth.png"
    out_pdf = OUT_DIR / f"profile_{profile_id}_{kd_name_tag}_compare_{depth_mode_tag}_depth.pdf"

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Single PNG saved to: {out_png}")
    print(f"Single PDF saved to: {out_pdf}")


# ============================================================
# 14. 组合图绘制函数
# ============================================================

def draw_profile_grid(
    df_all,
    profile_ids,
    profile_info,
    true_col,
    pred_col,
    baseline_col,
    kd_type,
    nrows=2,
    ncols=4
):
    if len(profile_ids) > nrows * ncols:
        raise ValueError("profile_ids 数量超过子图总数，请增大 nrows/ncols 或减少 profile_ids 数量。")

    if GRID_DEPTH_MODE not in ["global", "individual", "manual"]:
        raise ValueError("GRID_DEPTH_MODE 只能是 'global' / 'individual' / 'manual'。")

    share_y_flag = GRID_DEPTH_MODE in ["global", "manual"]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=GRID_FIGSIZE,
        sharey=share_y_flag
    )

    axes = np.array(axes).reshape(nrows, ncols)
    all_depths = []

    for idx, profile_id in enumerate(profile_ids):
        r = idx // ncols
        c = idx % ncols

        ax = axes[r, c]

        profile_df = get_profile_df(
            df_all=df_all,
            profile_id=profile_id,
            true_col=true_col,
            pred_col=pred_col
        )

        legend_loc, legend_bbox = get_grid_legend_position(
            idx=idx,
            profile_id=profile_id
        )

        result = plot_profile_on_ax(
            ax=ax,
            profile_df=profile_df,
            profile_id=profile_id,
            profile_info=profile_info,
            true_col=true_col,
            pred_col=pred_col,
            baseline_col=baseline_col,
            kd_type=kd_type,
            show_legend=True,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            legend_fontsize=GRID_LEGEND_FONTSIZE
        )

        if result is not None:
            depth_arr = np.asarray(result["depth"], dtype=float)
            depth_arr = depth_arr[np.isfinite(depth_arr)]

            if len(depth_arr) > 0:
                all_depths.extend(depth_arr)

                if GRID_DEPTH_MODE == "individual":
                    set_depth_ylim(
                        ax,
                        depth_values=depth_arr,
                        pad_ratio=DEPTH_PAD_RATIO
                    )

                print(
                    f"Profile {profile_id}: "
                    f"depth_min={np.nanmin(depth_arr):.3f}, "
                    f"depth_max={np.nanmax(depth_arr):.3f}, "
                    f"n={len(depth_arr)}"
                )

        if c == 0:
            ax.set_ylabel(
                "Depth (m)",
                fontsize=GRID_AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
                fontname=FONT_FAMILY
            )
        else:
            ax.set_ylabel("")

        if r == nrows - 1:
            ax.set_xlabel(
                rf"${get_kd_tex(kd_type)}$ (m$^{{-1}}$)",
                fontsize=GRID_AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
                fontname=FONT_FAMILY
            )
        else:
            ax.set_xlabel("")

    total_axes = nrows * ncols

    if len(profile_ids) < total_axes:
        for idx in range(len(profile_ids), total_axes):
            r = idx // ncols
            c = idx % ncols
            axes[r, c].set_visible(False)

    if GRID_DEPTH_MODE == "global":
        all_depths = np.asarray(all_depths, dtype=float)
        all_depths = all_depths[np.isfinite(all_depths)]

        if len(all_depths) == 0:
            raise ValueError("所有剖面都没有有效深度数据，无法设置 global 深度轴。")

        for ax in axes.flat:
            if ax.get_visible():
                set_depth_ylim(
                    ax,
                    depth_values=all_depths,
                    pad_ratio=DEPTH_PAD_RATIO
                )

    elif GRID_DEPTH_MODE == "manual":
        for ax in axes.flat:
            if ax.get_visible():
                set_depth_ylim(
                    ax,
                    manual_range=MANUAL_DEPTH_RANGE,
                    pad_ratio=DEPTH_PAD_RATIO
                )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97], h_pad=2.0, w_pad=1.5)

    profile_str = "_".join(map(str, profile_ids))
    kd_name_tag = get_kd_name_tag(kd_type)
    depth_mode_tag = GRID_DEPTH_MODE

    out_png = OUT_DIR / f"profiles_{profile_str}_grid_{nrows}x{ncols}_{kd_name_tag}_{depth_mode_tag}_depth.png"
    out_pdf = OUT_DIR / f"profiles_{profile_str}_grid_{nrows}x{ncols}_{kd_name_tag}_{depth_mode_tag}_depth.pdf"

    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Grid PNG saved to: {out_png}")
    print(f"Grid PDF saved to: {out_pdf}")


# ============================================================
# 15. 主程序
# ============================================================

def main():
    df_all = pd.read_csv(RECON_CSV)

    base_required_cols = ["profile_id", "depth"]

    for col in base_required_cols:
        if col not in df_all.columns:
            raise ValueError(f"重构文件缺少必要列：{col}")

    df_all = df_all.copy()
    df_all["profile_id"] = df_all["profile_id"].astype(int)
    df_all["depth"] = df_all["depth"].astype(float)

    TRUE_COL, PRED_COL, BASELINE_COL = resolve_kd_columns(df_all, KD_TYPE)
    profile_info = build_profile_info_from_mat(MAT_INFO_FILE, df_all)

    print("\n绘图配置：")
    print(f"PLOT_MODE        = {PLOT_MODE}")
    print(f"KD_TYPE          = {KD_TYPE}")
    print(f"RECON_CSV        = {RECON_CSV}")
    print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
    print(f"GRID_PROFILE_IDS = {GRID_PROFILE_IDS}")
    print(f"GRID_DEPTH_MODE  = {GRID_DEPTH_MODE}")
    print(f"GRID_FIGSIZE     = {GRID_FIGSIZE}")

    if PLOT_MODE == "single":
        draw_single_profile(
            df_all=df_all,
            profile_id=PROFILE_ID,
            profile_info=profile_info,
            true_col=TRUE_COL,
            pred_col=PRED_COL,
            baseline_col=BASELINE_COL,
            kd_type=KD_TYPE
        )

    elif PLOT_MODE == "grid":
        draw_profile_grid(
            df_all=df_all,
            profile_ids=GRID_PROFILE_IDS,
            profile_info=profile_info,
            true_col=TRUE_COL,
            pred_col=PRED_COL,
            baseline_col=BASELINE_COL,
            kd_type=KD_TYPE,
            nrows=GRID_NROWS,
            ncols=GRID_NCOLS
        )

    else:
        raise ValueError("PLOT_MODE 只能是 'single' 或 'grid'")


if __name__ == "__main__":
    main()