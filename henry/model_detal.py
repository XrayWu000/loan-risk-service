import joblib
import json
import pandas as pd
import lightgbm as lgb

# =========================
# 路徑
# =========================
MODEL_PATH = "lgbm_best_model.zip"
FEATURE_PATH = "lgbm_best_model_features.json"

# =========================
# 讀模型
# =========================
print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# =========================
# 讀特徵清單
# =========================
with open(FEATURE_PATH) as f:
    feature_list = json.load(f)

print("\nFeatures used:")
print(feature_list)

# =========================
# 基本資訊
# =========================
booster = model.booster_

print("\n========== 基本模型資訊 ==========")
print("Best iteration:", model.best_iteration_)
print("Number of trees:", booster.num_trees())
print("Number of features:", booster.num_feature())
print("Feature names:", booster.feature_name())
print("Objective:", booster.params.get("objective"))
print("Metric:", booster.params.get("metric"))
print("====================================")

# =========================
# 模型參數
# =========================
print("\n========== 模型參數 ==========")
for k, v in model.get_params().items():
    print(f"{k}: {v}")
print("================================")

# =========================
# Feature Importance
# =========================
importance_gain = booster.feature_importance(importance_type="gain")
importance_split = booster.feature_importance(importance_type="split")
feature_names = booster.feature_name()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "gain_importance": importance_gain,
    "split_importance": importance_split
}).sort_values(by="gain_importance", ascending=False)

print("\n========== Feature Importance (Gain) ==========")
print(importance_df)

# 儲存 feature importance
importance_df.to_csv("feature_importance.csv", index=False)

# =========================
# 取得完整模型結構
# =========================
model_structure = booster.dump_model()

print("\n========== 模型結構摘要 ==========")
print("Total trees:", len(model_structure["tree_info"]))

print("\nFirst tree structure preview:")
print(model_structure["tree_info"][0])

# =========================
# 匯出完整 JSON 結構
# =========================
with open("lgbm_full_model_structure.json", "w") as f:
    json.dump(model_structure, f, indent=2)

print("\nFull model structure saved to: lgbm_full_model_structure.json")
print("Feature importance saved to: feature_importance.csv")