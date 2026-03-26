import pandas as pd


# =========================
# 讀取資料
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# =========================
# 基本清理
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna()

    return df