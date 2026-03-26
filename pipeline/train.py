import lightgbm as lgb


def train_model(X, y, X_val=None, y_val=None):

    best_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "average_precision",
        "learning_rate": 0.05,

        "num_leaves": 31,
        "max_depth": 15,
        "min_child_samples": 40,

        "subsample": 0.6,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,

        "n_estimators": 20000,

        "class_weight": {0: 1, 1: 1.4},

        "random_state": 42,
        "n_jobs": -1,

        "lambda_l1": 12,
        "lambda_l2": 30,
    }

    model = lgb.LGBMClassifier(**best_params)

    if X_val is not None and y_val is not None:
        model.fit(
            X,
            y,
            eval_set=[(X, y), (X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=["person_home_ownership", "loan_intent", "person_gender"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(100)
            ],
        )
    else:
        model.fit(X, y)

    return model