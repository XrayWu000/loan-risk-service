import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_best_model.zip")
FEATURE_FILE = os.path.join(BASE_DIR, "models", "lgbm_best_model_features.json")
CSV_FILE = os.path.join(BASE_DIR, "data", "loan_requests_full.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "loan_train_36000.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "loan_test_9000.csv")

CSS_FILE = os.path.join(BASE_DIR, "frontend", "static", "styles.css")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_URL = f"{BASE_URL}/predict"

GENDER_OPTIONS = ["男", "女"]
EDUCATION_OPTIONS = ["高中/職", "副學士(專科)", "學士", "碩士", "博士"]
HOME_OWNERSHIP_OPTIONS = ["租賃", "自有（尚有貸款）", "自有（無貸款）"]
LOAN_INTENT_OPTIONS = ["個人周轉", "教育進修", "醫療照護", "創業周轉"]
