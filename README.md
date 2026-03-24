# Loan Risk Assessment System (Machine Learning)

## 📌 Project Overview（專題簡介）
本專題為一個完整的 **信用貸款風險預測系統（Loan Default Classification）**，  
透過機器學習模型預測申請人是否會發生違約（loan_status）。

專案涵蓋從資料處理、特徵工程、模型訓練，到模型解釋與應用部署的完整流程。

---

## 🎯 Project Objective（專案目標）
- 建立一個可預測貸款違約風險的分類模型
- 提升模型在風險控管上的 **Recall（召回率）**
- 透過 Explainable AI（SHAP）強化模型可解釋性
- 將模型整合為可實際使用的應用系統

---

## 📊 Dataset（資料集說明）
- Training Data：36,000 筆
- Test Data：9,000 筆
- 主要特徵包含：

### 1️⃣ 人口資訊（Demographic）
- person_age（年齡）
- person_gender（性別）
- person_education（教育程度）

### 2️⃣ 財務資訊（Financial）
- person_income（年收入）
- person_emp_exp（工作年資）
- person_home_ownership（房屋狀況）

### 3️⃣ 貸款資訊（Loan）
- loan_amnt（貸款金額）
- loan_intent（貸款用途）
- loan_int_rate（利率）
- loan_percent_income（收入負擔比）

### 4️⃣ 信用資訊（Credit）
- credit_score（信用分數）
- cb_person_cred_hist_length（信用歷史長度）

---

## ⚙️ Machine Learning Pipeline（模型流程）

1. Data Cleaning（資料清洗）
2. Feature Engineering（特徵工程）
3. Model Training（模型訓練）
4. Model Evaluation（模型評估）
5. Model Interpretation（SHAP）
6. Deployment（Gradio 應用）

---

## 🧠 Feature Engineering（特徵工程）

- log_income = log1p(person_income)
- log_loan_amnt = log1p(loan_amnt)
- interest_pressure = loan_int_rate × loan_percent_income
- debt_ratio = loan_amnt / income
- credit_score_bucket（信用分群）
- age_bucket（年齡分群）

---

## 🤖 Models（模型）

本專案比較多種模型，最終選擇：

### ✅ LightGBM（最佳模型）
原因：
- 高效能（High Performance）
- 支援類別特徵（Categorical Features）
- 易於解釋（搭配 SHAP）

---

## 📈 Model Performance（模型表現）

| Metric | Score |
|------|------|
| Accuracy | ~0.97 |
| Precision | ~0.95 |
| Recall | ~0.93 |
| AUC | ~0.98 |

👉 本專案重點在 **Recall（降低違約風險漏判）**

---

## 🔍 Explainable AI（模型解釋）

使用 **SHAP（SHapley Additive exPlanations）**：

- 分析特徵重要度
- 解釋單筆預測（waterfall plot）
- 提供商業決策依據

---

## 💻 Application（應用系統）

使用 **Gradio** 建立 AI Web App：

功能：
- 使用者輸入貸款資訊
- 即時預測違約機率
- 顯示風險等級：
  - ✅ 低風險（核貸）
  - ⚠️ 中風險（人工審核）
  - 🚨 高風險（拒絕）

---

## 📂 Project Structure
---
loan-approval-ml/
│
├─ henry/ # 模型開發、特徵工程、Gradio應用
├─ ring/ # SHAP分析與模型實驗
├─ kershaw/ # 特徵工程與EDA
├─ hsinyi/ # 模型評估與比較
├─ Mike/ # 資料處理與分析
│
├─ README.md
└─ .gitignore

---

## 🚀 Key Highlights（專案亮點）

- 完整 ML Pipeline（資料 → 模型 → 應用）
- 強調 **Recall optimization（風險控制導向）**
- 使用 **SHAP 提升模型可解釋性**
- 成功部署為 **可操作的 AI Web App**

---

## 🔮 Future Work（未來優化）

- API 化（FastAPI / Flask）
- Docker 部署
- 自動化 Data Pipeline
- 模型監控（Model Monitoring）

---

## 👥 Team Members

- Henry（Model & System Integration）
- Ring（SHAP & Analysis）
- Kershaw（Feature Engineering）
- Hsinyi（Model Evaluation）
- Mike（Data Processing）
