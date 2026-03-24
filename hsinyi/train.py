import matplotlib

matplotlib.use('Agg')

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def train_lgbm_model():
    # 1. 載入資料
    file_path = 'loan_train_36000_engineered.csv'
    print(f"正在載入資料: {file_path} ...")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {file_path}。請確認檔案是否在同一資料夾內。")
        return

    # 2. 準備特徵與目標
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 3. 切分資料
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 建立模型
    print("開始訓練 LightGBM 模型... (請稍候)")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',  # 維持加權，抓壞人效果好
        n_jobs=-1
    )

    # 5. 訓練
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # 6. 評估與繪圖
    print("\n--- 模型評估結果 ---")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # 基本指標
    print(f"準確率 (Accuracy): {accuracy_score(y_val, y_pred):.4f}")
    print(f"AUC Score       : {roc_auc_score(y_val, y_prob):.4f}")
    from sklearn.metrics import f1_score
    print(f"F1-score (類別1) : {f1_score(y_val, y_pred):.4f}")
    print("\n詳細分類報告:")
    print(classification_report(y_val, y_pred))

    # 繪製混淆矩陣
    print("正在繪製混淆矩陣...")
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal(0)', 'Default(1)'])

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Balanced Weight)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_result.png')  # 只存檔
    print("混淆矩陣已儲存為: confusion_matrix_result.png")


    # 7. 繪製特徵重要性
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', title='Feature Importance (Gain)',
                        height=0.5)
    plt.tight_layout()
    plt.savefig('feature_importance_result.png')  # 只存檔
    print("特徵重要性圖表已儲存為: feature_importance_result.png")

    # 8. 儲存模型 (改為壓縮格式)
    # compress=3 代表壓縮層級 (1-9)，3 是速度與大小的最佳平衡
    model_filename = 'lgbm_loan_model.zip'
    joblib.dump(model, model_filename, compress=3)
    print(f"\n模型已壓縮並儲存為 {model_filename}")

if __name__ == "__main__":
    train_lgbm_model()