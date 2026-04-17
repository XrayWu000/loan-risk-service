from sklearn.model_selection import train_test_split

from config.path_config import FEATURE_FILE, MODEL_PATH, TRAIN_PATH
from pipeline.data_loader import clean_data, load_data
from pipeline.evaluate import evaluate_model
from pipeline.feature_engineering import prepare_model_input
from pipeline.save import save_features, save_model
from pipeline.train import train_model


def main():
    print("Loading training data...")
    df = load_data(TRAIN_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Preparing features...")
    X = prepare_model_input(df)
    y = df["loan_status"]

    print("Training model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train, X_val, y_val)

    print("Evaluating model...")
    evaluate_model(model, X, y, name="Training")

    print("Saving model...")
    save_model(model, MODEL_PATH)

    print("Saving feature list...")
    save_features(list(X.columns), FEATURE_FILE)

    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
