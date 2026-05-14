import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from pygam import LinearGAM, s, te
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 05_train_gam_kd_top_valid_soft.py
# 基于“最顶端有效数据”的软分段 GAM 建模
# 输出统一列名：pred_mid / pred_log_kd / pred_kd
# =========================

# -------------------------------------------------
# 1. 路径设置
# -------------------------------------------------
train_file = r"D:\work\train_split.csv"
test_file = r"D:\work\test_split.csv"
config_file = r"D:\work\gam_config.json"

output_dir = r"D:\work"
model_dir = r"D:\work\models"
log_dir = r"D:\work\logs"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# -------------------------------------------------
# 2. 读取配置
# -------------------------------------------------
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)

active_feature_set = config["active_feature_set"]
target_col = config.get("target_col", "residual")
feature_sets = config["feature_sets"]
derived_features_cfg = config.get("derived_features", {})
soft_cfg = config.get("soft_partition", {})
n_splines_cfg = config.get("n_splines", {})

if active_feature_set not in feature_sets:
    raise ValueError(f"active_feature_set 不存在: {active_feature_set}")

feature_cols = feature_sets[active_feature_set]

soft_enabled = soft_cfg.get("enabled", True)
z0 = float(soft_cfg.get("z0", 10.0))
tau = float(soft_cfg.get("tau", 2.0))
weight_floor = float(soft_cfg.get("weight_floor", 0.02))

shallow_use_depth_chl_interaction = bool(
    soft_cfg.get("shallow_use_depth_chl_interaction", True)
)
deep_use_depth_chl_interaction = bool(
    soft_cfg.get("deep_use_depth_chl_interaction", False)
)

print("当前配置文件：", config_file)
print("active_feature_set =", active_feature_set)
print("target_col =", target_col)
print("feature_cols =", feature_cols)
print("soft_enabled =", soft_enabled)
print("z0 =", z0)
print("tau =", tau)
print("weight_floor =", weight_floor)
print("shallow_use_depth_chl_interaction =", shallow_use_depth_chl_interaction)
print("deep_use_depth_chl_interaction =", deep_use_depth_chl_interaction)

# -------------------------------------------------
# 3. 读取训练集和测试集
# -------------------------------------------------
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

print("\n训练集维度：", train_data.shape)
print("测试集维度：", test_data.shape)

# -------------------------------------------------
# 4. 必要列检查
# -------------------------------------------------
required_cols = [
    "profile_id", "depth", "Kd_slow",
    "log_Kd_slow", "log_Kd_baseline", "residual",
    "top_valid_depth", "top_chl", "top_temperature", "top_salinity"
]

missing_cols_train = [c for c in required_cols if c not in train_data.columns]
missing_cols_test = [c for c in required_cols if c not in test_data.columns]

if missing_cols_train:
    raise ValueError(f"train_data 缺少必要列: {missing_cols_train}")
if missing_cols_test:
    raise ValueError(f"test_data 缺少必要列: {missing_cols_test}")

print("\n必要列检查通过")

# -------------------------------------------------
# 5. 派生特征
# -------------------------------------------------
use_log_top_chl = derived_features_cfg.get("use_log_top_chl", False)

for df in [train_data, test_data]:
    if use_log_top_chl:
        df["log_top_chl"] = np.log(df["top_chl"] + 1e-6)

# -------------------------------------------------
# 6. 检查当前特征组合中的列是否存在
# -------------------------------------------------
missing_feature_cols = [col for col in feature_cols if col not in train_data.columns]
if missing_feature_cols:
    raise ValueError(f"当前特征组合缺少列: {missing_feature_cols}")

# -------------------------------------------------
# 7. 构造输入输出
# -------------------------------------------------
X_train = train_data[feature_cols].copy()
y_train = train_data[target_col].copy()

X_test = test_data[feature_cols].copy()
y_test = test_data[target_col].copy()

print("\n本次真正用于训练的 X_train.columns：")
print(X_train.columns.tolist())

assert X_train.columns.tolist() == feature_cols, "X_train.columns 和 feature_cols 不一致！"

X_train_np = X_train.to_numpy(dtype=float)
y_train_np = y_train.to_numpy(dtype=float)
X_test_np = X_test.to_numpy(dtype=float)
y_test_np = y_test.to_numpy(dtype=float)

# -------------------------------------------------
# 8. 软分段深度权重函数
# -------------------------------------------------
def depth_weight_shallow(depth_array, z0=10.0, tau=2.0, weight_floor=0.02):
    """
    浅层权重：
        w_s(z) = 1 / (1 + exp((z - z0)/tau))
    tau 越小，过渡越陡；tau 越大，过渡越平滑
    """
    depth_array = np.asarray(depth_array, dtype=float)
    tau = max(float(tau), 1e-6)

    w = 1.0 / (1.0 + np.exp((depth_array - z0) / tau))

    # 防止权重过于接近0或1导致数值不稳定
    if weight_floor is not None:
        weight_floor = float(weight_floor)
        weight_floor = min(max(weight_floor, 0.0), 0.49)
        w = np.clip(w, weight_floor, 1.0 - weight_floor)

    return w

# 训练集和测试集的融合权重
train_data["weight_shallow"] = depth_weight_shallow(
    train_data["depth"].values, z0=z0, tau=tau, weight_floor=weight_floor
)
test_data["weight_shallow"] = depth_weight_shallow(
    test_data["depth"].values, z0=z0, tau=tau, weight_floor=weight_floor
)

train_data["weight_deep"] = 1.0 - train_data["weight_shallow"]
test_data["weight_deep"] = 1.0 - test_data["weight_shallow"]

# -------------------------------------------------
# 9. 自动构造 GAM 项
# -------------------------------------------------
def build_gam_terms(feature_cols, n_splines_cfg, use_depth_chl_interaction=True):
    """
    根据当前 feature_cols 自动构造 GAM 项

    规则：
    1. 每个特征都加一个 s(i)
    2. 若同时有 depth 和 top_chl，则可加 te(depth, top_chl)
    3. 若同时有 depth 和 log_top_chl，则可加 te(depth, log_top_chl)
    4. 自动保证 n_splines > spline_order
    """
    if len(feature_cols) == 0:
        raise ValueError("feature_cols 不能为空")

    spline_order = 3
    min_valid_splines = spline_order + 1  # 至少 4

    col_idx = {col: i for i, col in enumerate(feature_cols)}
    terms = None

    for i, col in enumerate(feature_cols):
        raw_n_splines = n_splines_cfg.get(col, 10)
        n_splines = max(int(raw_n_splines), min_valid_splines)

        if raw_n_splines != n_splines:
            print(f"[警告] {col} 的 n_splines={raw_n_splines} 非法，已自动修正为 {n_splines}")

        term = s(i, n_splines=n_splines, spline_order=spline_order)
        terms = term if terms is None else terms + term

    if use_depth_chl_interaction and "depth" in col_idx:
        raw_interaction = n_splines_cfg.get("interaction", [6, 6])

        if not isinstance(raw_interaction, (list, tuple)) or len(raw_interaction) != 2:
            raise ValueError(f"interaction 必须是长度为2的列表，例如 [6, 6]，当前为: {raw_interaction}")

        interaction_splines = [
            max(int(raw_interaction[0]), min_valid_splines),
            max(int(raw_interaction[1]), min_valid_splines)
        ]

        if "top_chl" in col_idx:
            terms = terms + te(
                col_idx["depth"],
                col_idx["top_chl"],
                n_splines=interaction_splines,
                spline_order=[spline_order, spline_order]
            )
        elif "log_top_chl" in col_idx:
            terms = terms + te(
                col_idx["depth"],
                col_idx["log_top_chl"],
                n_splines=interaction_splines,
                spline_order=[spline_order, spline_order]
            )

    return terms

# -------------------------------------------------
# 10. 建立两个 GAM：浅层模型 + 深层模型
# -------------------------------------------------
shallow_terms = build_gam_terms(
    feature_cols=feature_cols,
    n_splines_cfg=n_splines_cfg,
    use_depth_chl_interaction=shallow_use_depth_chl_interaction
)

deep_terms = build_gam_terms(
    feature_cols=feature_cols,
    n_splines_cfg=n_splines_cfg,
    use_depth_chl_interaction=deep_use_depth_chl_interaction
)

print("\n浅层 GAM 结构：")
print(shallow_terms)
print("\n深层 GAM 结构：")
print(deep_terms)

shallow_gam = LinearGAM(shallow_terms)
deep_gam = LinearGAM(deep_terms)

print("\n浅层模型对象：")
print(shallow_gam)
print("\n深层模型对象：")
print(deep_gam)

# -------------------------------------------------
# 11. 训练两个 GAM
# 通过 sample weights 实现软分段
# -------------------------------------------------
if soft_enabled:
    shallow_train_weights = train_data["weight_shallow"].values
    deep_train_weights = train_data["weight_deep"].values
else:
    shallow_train_weights = np.ones(len(train_data), dtype=float)
    deep_train_weights = np.ones(len(train_data), dtype=float)

print("\n开始训练浅层模型...")
shallow_gam.gridsearch(
    X_train_np,
    y_train_np,
    weights=shallow_train_weights,
    progress=True
)
print("浅层模型训练完成")

print("\n开始训练深层模型...")
deep_gam.gridsearch(
    X_train_np,
    y_train_np,
    weights=deep_train_weights,
    progress=True
)
print("深层模型训练完成")

# -------------------------------------------------
# 12. 保存两个模型
# -------------------------------------------------
shallow_model_path = os.path.join(model_dir, f"gam_shallow_{active_feature_set}.joblib")
deep_model_path = os.path.join(model_dir, f"gam_deep_{active_feature_set}.joblib")

joblib.dump(shallow_gam, shallow_model_path)
joblib.dump(deep_gam, deep_model_path)

print("\n训练好的模型已保存：")
print(shallow_model_path)
print(deep_model_path)

# -------------------------------------------------
# 13. 预测
# 分别预测，再按深度权重融合
# -------------------------------------------------
train_data["pred_mid_shallow"] = shallow_gam.predict(X_train_np)
train_data["pred_mid_deep"] = deep_gam.predict(X_train_np)

test_data["pred_mid_shallow"] = shallow_gam.predict(X_test_np)
test_data["pred_mid_deep"] = deep_gam.predict(X_test_np)

train_data["pred_mid"] = (
    train_data["weight_shallow"] * train_data["pred_mid_shallow"] +
    train_data["weight_deep"] * train_data["pred_mid_deep"]
)

test_data["pred_mid"] = (
    test_data["weight_shallow"] * test_data["pred_mid_shallow"] +
    test_data["weight_deep"] * test_data["pred_mid_deep"]
)

# 兼容旧列
if target_col == "residual":
    train_data["residual_pred_gam"] = train_data["pred_mid"]
    test_data["residual_pred_gam"] = test_data["pred_mid"]
elif target_col == "log_Kd_slow":
    train_data["log_Kd_direct_pred_gam"] = train_data["pred_mid"]
    test_data["log_Kd_direct_pred_gam"] = test_data["pred_mid"]

# -------------------------------------------------
# 14. 评估函数
# -------------------------------------------------
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_metrics(y_true, y_pred, name="dataset"):
    mae, rmse, r2 = calc_metrics(y_true, y_pred)
    print(f"\n{name} 指标：")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R2   = {r2:.6f}")

# -------------------------------------------------
# 15. 中间目标评估
# -------------------------------------------------
print_metrics(train_data[target_col], train_data["pred_mid"], name="Train mid target (soft GAM)")
print_metrics(test_data[target_col], test_data["pred_mid"], name="Test mid target (soft GAM)")

# -------------------------------------------------
# 16. 恢复到最终 Kd_slow
# -------------------------------------------------
if target_col == "residual":
    train_data["pred_log_kd"] = train_data["log_Kd_baseline"] + train_data["pred_mid"]
    test_data["pred_log_kd"] = test_data["log_Kd_baseline"] + test_data["pred_mid"]
elif target_col == "log_Kd_slow":
    train_data["pred_log_kd"] = train_data["pred_mid"]
    test_data["pred_log_kd"] = test_data["pred_mid"]
else:
    raise ValueError(f"当前 target_col={target_col} 暂不支持后处理")

train_data["pred_kd"] = np.exp(train_data["pred_log_kd"])
test_data["pred_kd"] = np.exp(test_data["pred_log_kd"])

# 兼容旧列
train_data["log_Kd_pred_gam"] = train_data["pred_log_kd"]
test_data["log_Kd_pred_gam"] = test_data["pred_log_kd"]

train_data["Kd_pred_gam"] = train_data["pred_kd"]
test_data["Kd_pred_gam"] = test_data["pred_kd"]

# -------------------------------------------------
# 17. 最终任务评估
# -------------------------------------------------
print_metrics(train_data["Kd_slow"], train_data["pred_kd"], name="Train Kd_slow (soft GAM)")
print_metrics(test_data["Kd_slow"], test_data["pred_kd"], name="Test Kd_slow (soft GAM)")

# -------------------------------------------------
# 18. 查看结果
# -------------------------------------------------
print("\n测试集预测结果前10行：")
print(
    test_data[
        [
            "profile_id", "depth",
            "Kd_slow", "pred_kd",
            "log_Kd_slow", "pred_log_kd",
            target_col, "pred_mid",
            "weight_shallow", "weight_deep"
        ]
    ].head(10)
)

# -------------------------------------------------
# 19. 保存结果
# -------------------------------------------------
train_pred_file = os.path.join(output_dir, f"train_with_pred_soft_gam_{active_feature_set}.csv")
test_pred_file = os.path.join(output_dir, f"test_with_pred_soft_gam_{active_feature_set}.csv")

train_data.to_csv(train_pred_file, index=False, encoding="utf-8-sig")
test_data.to_csv(test_pred_file, index=False, encoding="utf-8-sig")

print("\n已保存：", train_pred_file)
print("已保存：", test_pred_file)

# -------------------------------------------------
# -------------------------------------------------
# 20. 保存实验摘要，并追加到 experiment_log.csv
# -------------------------------------------------
train_mae, train_rmse, train_r2 = calc_metrics(train_data["Kd_slow"], train_data["pred_kd"])
test_mae, test_rmse, test_r2 = calc_metrics(test_data["Kd_slow"], test_data["pred_kd"])

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_row = {
    "run_id": run_id,
    "run_time": run_time,
    "model_class": "SoftPartitionGAM",
    "note": f"soft_gam {active_feature_set}",

    "feature_cols": json.dumps(feature_cols, ensure_ascii=False),
    "target_col": target_col,

    "n_train_rows": int(len(train_data)),
    "n_test_rows": int(len(test_data)),
    "n_train_profiles": int(train_data["profile_id"].nunique()),
    "n_test_profiles": int(test_data["profile_id"].nunique()),

    "soft_enabled": soft_enabled,
    "z0": z0,
    "tau": tau,
    "weight_floor": weight_floor,
    "shallow_use_depth_chl_interaction": shallow_use_depth_chl_interaction,
    "deep_use_depth_chl_interaction": deep_use_depth_chl_interaction,

    "train_final_mae": float(train_mae),
    "train_final_rmse": float(train_rmse),
    "train_final_r2": float(train_r2),
    "test_final_mae": float(test_mae),
    "test_final_rmse": float(test_rmse),
    "test_final_r2": float(test_r2),

    "shallow_model_file": shallow_model_path,
    "deep_model_file": deep_model_path,
    "train_pred_file": train_pred_file,
    "test_pred_file": test_pred_file
}

log_df = pd.DataFrame([log_row])

csv_log_file = os.path.join(log_dir, "experiment_log.csv")

if os.path.exists(csv_log_file):
    old_log = pd.read_csv(csv_log_file)
    new_log = pd.concat([old_log, log_df], ignore_index=True)
else:
    new_log = log_df.copy()

new_log.to_csv(csv_log_file, index=False, encoding="utf-8-sig")

# 同时保留单独 json
summary = log_row.copy()
summary_file = os.path.join(log_dir, f"soft_gam_summary_{active_feature_set}_{run_id}.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n实验记录已保存：")
print(csv_log_file)
print(summary_file)