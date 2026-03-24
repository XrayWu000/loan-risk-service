import matplotlib

matplotlib.use('Agg')

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def validate_9000_model():
    # 1. 設定檔案路徑
    test_file = 'loan_test_9000_engineered.csv'
    model_file = 'lgbm_loan_model.zip'

    print(f"--- 開始驗證 ---")

    # 2. 載入 9000 筆測試資料
    try:
        df_test = pd.read_csv(test_file)
        print(f"成功載入: {test_file}")
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {test_file}，請確認檔案已放入資料夾。")
        return

    # 3. 載入訓練好的模型
    try:
        model = joblib.load(model_file)
        print(f"成功載入模型: {model_file}")
    except FileNotFoundError:
        print(f"錯誤: 找不到模型檔 {model_file}，請先執行訓練程式。")
        return

    # 4. 準備特徵 (X) 與標籤 (y)
    X_test = df_test.drop(columns=['loan_status'])
    y_test = df_test['loan_status']

    # 5. 執行預測
    print("正在進行預測...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. 輸出報告
    print("\n" + "=" * 45)
    print("      Final Test Report (9,000 Samples)")
    print("=" * 45)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"準確率 (Accuracy) : {acc:.4f}")
    print(f"AUC Score        : {auc:.4f}")
    print(f"F1-score (類別1)  : {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Default']))

    # 7. 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Default'])

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Greens', values_format='d')
    plt.title('Confusion Matrix - 9,000 Samples Test')
    plt.tight_layout()

    output_img = 'validation_9000_result.png'
    plt.savefig(output_img)
    print(f"\n驗證圖表已儲存為: {output_img}")
    print("=" * 45)


if __name__ == "__main__":
    validate_9000_model()