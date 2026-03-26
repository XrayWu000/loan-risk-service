import joblib
import json
import os


# =========================
# 存模型
# =========================
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


# =========================
# 存 feature list
# =========================
def save_features(features, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    print(f"✅ Features saved to {path}")