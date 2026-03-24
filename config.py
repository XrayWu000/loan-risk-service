import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_best_model.zip")
FEATURE_FILE = os.path.join(BASE_DIR, "lgbm_best_model_features.json")
CSV_FILE = os.path.join(BASE_DIR, "data", "loan_requests_full.csv")

CSS_FILE = os.path.join(BASE_DIR, "frontend", "static", "styles.css")

API_URL = "http://127.0.0.1:8000/predict"