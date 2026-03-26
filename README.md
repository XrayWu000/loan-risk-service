# 🏦 Loan Risk Assessment System

A production-oriented end-to-end machine learning system for loan default prediction, including model training, API deployment, and interactive user interface.

---

## 🚀 Live Demo

- 🌐 Web App: http://localhost:7860
- 📘 API Docs: http://localhost:8000/docs

---

## 📌 Project Overview

This project builds a **Loan Risk Assessment System** to predict the probability of loan default based on applicant information.

The system is designed as a complete AI application, not just a machine learning model.

---

## 💼 Use Case

This system simulates a real-world bank loan approval workflow:

1. User submits loan application
2. System evaluates risk using ML model
3. Outputs decision:
   - Approve
   - Manual Review
   - Reject

Business value:

- Reduce default risk
- Improve decision efficiency
- Provide interpretable AI decisions

---

## 🧠 Machine Learning

- Model: LightGBM
- Task: Binary Classification (Default / Non-default)
- Dataset: ~45,000 loan records

### Feature Engineering

- log transformation: `log_income`
- interaction feature: `interest_pressure`
- categorical encoding (manual mapping)

### Training Strategy

- Metric: Average Precision (PR-AUC)
- Class Weighting: Improve recall for high-risk cases
- Early Stopping: Prevent overfitting
- Regularization: L1 / L2 applied

---

## ⚖️ Decision Strategy

Based on predicted probability:

- ≥ 0.50 → 🚨 Reject (High Risk)
- ≥ 0.20 → ⚠️ Manual Review (Medium Risk)
- < 0.20 → ✅ Approve (Low Risk)

---

## 🏗️ System Architecture

```mermaid
flowchart TB

A[User Input]
→ B[Gradio UI]
→ C[FastAPI API]
→ D[Feature Engineering Pipeline]
→ E[LightGBM Model]
→ F[Prediction Result]

F → G[Decision Logic]
G → H[UI Display]

F → I[CSV Logging]
I → J[Dashboard]
```

---

## 🔄 Machine Learning Pipeline

```mermaid
flowchart TB

A[Raw Data CSV]
→ B[Data Loading]
→ C[Feature Engineering]
→ D[Train / Validation Split]
→ E[Model Training]
→ F[Evaluation]
→ G[Save Model & Features]

G → H[Inference Pipeline]
```

---

## ⚙️ API Features

- Input validation (Pydantic)
- Error handling (HTTPException)
- Unified feature engineering pipeline
- Stable response format

### Example Response

```
{
  "status": "success",
  "probability": 0.23
}
```

---

## 💻 Frontend (Gradio)

- User-friendly loan application UI
- Real-time prediction
- Input validation
- Risk decision output

---

## 🗂️ Logging System

- Stores all prediction requests in CSV
- Includes:
  - Input data
  - Prediction probability
  - Timestamp

- Enables:
  - Monitoring
  - Future retraining
  - Data analysis

---

## 📁 Project Structure

```
loan-risk-service/
├── app/                  # FastAPI backend
├── frontend/             # Gradio UI
├── pipeline/             # ML pipeline
├── models/               # trained model + features
├── data/                 # logs / dataset
├── config.py             # central config
├── main_train.py         # training entry
├── docker-compose.yml
```

---

## 🐳 Deployment

Run the system locally:

```
docker compose up --build
```

Access:

- API → http://localhost:8000/docs
- UI → http://localhost:7860

---

## 🔧 Tech Stack

- Python
- LightGBM
- FastAPI
- Gradio
- Pandas / NumPy
- Docker

---

## 🚀 Key Highlights

- End-to-end ML system (training → deployment → UI)
- Consistent feature pipeline (train = inference)
- Production-oriented design (API + logging)
- Decision-focused ML (threshold tuning for recall)

---

## 📈 Future Improvements

- Replace CSV with database (PostgreSQL)
- Model monitoring (data drift / performance)
- Cloud deployment (GCP / AWS / Render)
- Authentication system

---

## 👨‍💻 Author

Henry Wu