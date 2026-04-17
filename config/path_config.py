from config.base_config import BASE_DIR, get_path

MODEL_PATH = get_path(
    "MODEL_PATH",
    BASE_DIR / "models" / "lgbm_best_model.zip",
)

FEATURE_FILE = get_path(
    "FEATURE_FILE",
    BASE_DIR / "models" / "lgbm_best_model_features.json",
)

LOCAL_LOG_PATH = get_path(
    "LOCAL_LOG_PATH",
    BASE_DIR / "data" / "loan_requests_full.csv",
)
CSV_FILE = LOCAL_LOG_PATH

TRAIN_PATH = get_path(
    "TRAIN_PATH",
    BASE_DIR / "data" / "loan_train_36000.csv",
)
TRAIN_DATA_PATH = TRAIN_PATH

TEST_PATH = get_path(
    "TEST_PATH",
    BASE_DIR / "data" / "loan_test_9000.csv",
)

CSS_FILE = get_path(
    "CSS_FILE",
    BASE_DIR / "frontend" / "static" / "styles.css",
)
