# 🏦 Loan Risk Assessment System

A complete end-to-end machine learning system for loan default prediction, including model training, API deployment, and interactive user interface.

---

## 📌 Project Overview

This project builds a **Loan Risk Assessment System** using machine learning to predict the probability of loan default based on applicant information.

The system integrates:

- Machine Learning Model (LightGBM)
- FastAPI (Model Serving)
- Gradio (User Interface)
- CSV Logging System
- Docker (Deployment)

---

## 🎯 Objectives

- Predict loan default risk using structured financial data
- Build an end-to-end AI application (not just a model)
- Provide interpretable and deployable solution

---

## 🧠 Machine Learning

- Model: LightGBM
- Task: Binary Classification (Default / Non-default)
- Dataset: ~45,000 loan records
- Features include:
  - Demographics (age, gender)
  - Financial info (income, employment)
  - Loan details (amount, interest rate)

### Feature Engineering

- Log transformation: `log_income`
- Interaction feature: `interest_pressure`
- Categorical encoding (manual mapping)

---

## 🏗️ System Architecture
```
User Input (Gradio UI)
↓
Frontend (Gradio)
↓ API Request
FastAPI Backend
↓
Model Inference (LightGBM)
↓
Prediction Result
↓
CSV Logging + Dashboard
```
---

## 📁 Project Structure
```
loan-risk-service/
├── app/ # FastAPI backend
├── frontend/ # Gradio UI
│ ├── services/ # logging & dashboard
│ └── static/ # CSS
├── models/ # trained model
├── data/ # CSV logs
├── config.py # central config
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
└── requirements.txt
```
## ⚙️ API Features

- Input validation (Pydantic)
- Error handling (HTTPException)
- Feature preprocessing pipeline
- Stable response format

### Example Response

```
json
{
  "status": "success",
  "probability": 0.23
}
```
## 💻 Frontend Features (Gradio)

- User-friendly loan application UI
- Real-time prediction
- Input validation
- CSV logging system
- Dashboard (view latest records)
- Download report functionality

---

## 🗂️ Logging System

- Saves all requests to CSV
- Includes:
  - Input data
  - Prediction result
  - Timestamp
- Thread-safe writing (lock mechanism)
- Retry mechanism for write failures

---

## 🐳 Deployment (Docker)

### Run the system

```
bash
docker compose up --build
```
### Access services

- API: http://localhost:8000/docs
- UI: http://localhost:7860

---

## 🔧 Tech Stack

- Python
- FastAPI
- Gradio
- LightGBM
- Pandas / NumPy
- Docker

---

## 🚀 Key Highlights

- End-to-end ML system (not just model)
- Modular architecture (frontend / backend / service layer)
- Production-oriented design (API + validation + logging)
- Dockerized deployment

---

## 📈 Future Improvements

- Add authentication system
- Replace CSV with database (PostgreSQL)
- Deploy to cloud (GCP / AWS / Render)
- Add model monitoring

---

## 👨‍💻 Author

Henry Wu

---

## 🔗 Demo

- API Docs: http://localhost:8000/docs
- UI: http://localhost:7860