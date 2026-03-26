import os
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.data_loader import load_data, clean_data
from pipeline.feature_engineering import prepare_model_input
from pipeline.train import train_model
from pipeline.evaluate import evaluate
from pipeline.save import save_model, save_features

from config import MODEL_PATH, FEATURE_FILE, TRAIN_DATA_PATH


def main():
    print("🔹 Loading data...")
    df = load_data(TRAIN_DATA_PATH)

    print("🔹 Cleaning data...")
    df = clean_data(df)

    print("🔹 Preparing features...")
    X = prepare_model_input(df)
    y = df["loan_status"]

    print("🔹 Training model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train, X_val, y_val)

    print("🔹 Evaluating model...")
    evaluate(model, X, y)

    print("🔹 Saving model...")
    save_model(model, MODEL_PATH)

    print("🔹 Saving feature list...")
    save_features(
        list(X.columns),
        FEATURE_FILE
    )

    print("✅ Training pipeline completed!")


if __name__ == "__main__":
    main()