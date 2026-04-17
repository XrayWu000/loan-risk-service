import json
import numpy as np
import pandas as pd

from config.path_config import FEATURE_FILE


def get_feature_columns():
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_categorical_cols():
    return ["person_home_ownership", "loan_intent", "person_gender"]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gender_map = {
        "男": 1,
        "女": 0,
        "male": 1,
        "female": 0,
    }

    home_map = {
        "租賃": 0,
        "自有（尚有貸款）": 1,
        "自有（無貸款）": 2,
        "RENT": 0,
        "MORTGAGE": 1,
        "OWN": 2,
    }

    intent_map = {
        "個人周轉": 0,
        "醫療照護": 1,
        "創業周轉": 2,
        "教育進修": 3,
        "PERSONAL": 0,
        "MEDICAL": 1,
        "VENTURE": 2,
        "EDUCATION": 3,
    }

    if "person_gender" in df.columns:
        df["person_gender"] = df["person_gender"].map(gender_map)

    if "person_home_ownership" in df.columns:
        df["person_home_ownership"] = df["person_home_ownership"].map(home_map)

    if "loan_intent" in df.columns:
        df["loan_intent"] = df["loan_intent"].map(intent_map)

    df["log_income"] = np.log1p(df["person_income"])
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

    return df


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    df = feature_engineering(df)

    feature_columns = get_feature_columns()
    df = df[feature_columns]

    for col in ["person_home_ownership", "loan_intent", "person_gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
