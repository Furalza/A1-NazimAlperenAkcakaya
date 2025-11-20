import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from NeuralNet import NeuralNet

RANDOM_STATE = 42


# ------------------ Metrics ------------------

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return mse, mae, mape


# ------------------ Data loader ------------------

def load_bike_data(path="data/day.csv"):
    df = pd.read_csv(path)

    drop_cols = ["instant", "dteday", "casual", "registered"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df["cnt"].astype(float).values
    X = df.drop(columns=["cnt"])

    cat_cols = ["season", "yr", "mnth", "holiday", "weekday",
                "workingday", "weathersit"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols


# ------------------ Preprocessing ------------------

def train_test_split_bike(test_size=0.2):
    X, y, num_cols, cat_cols = load_bike_data()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
    )

    X_train_full_proc = preprocessor.fit_transform(X_train_full)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_full_proc, X_test_proc, y_train_full, y_test, preprocessor


# ------------------ K-fold CV for BP ------------------

def kfold_cv_neuralnet(X, y, layers, n_epochs, lr, mom, activation, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    mse_scores, mae_scores, mape_scores = [], [], []

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_s, y_test = y_scaled[train_idx], y[test_idx]

        nn = NeuralNet(
            layers=layers,
            n_epochs=n_epochs,
            learning_rate=lr,
            momentum=mom,
            activation=activation,
            val_ratio=0.2,
            random_state=RANDOM_STATE,
        )
        nn.fit(X_train, y_train_s)

        pred_scaled = nn.predict(X_test)
        pred_real = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()

        mse, mae, mape = regression_metrics(y_test, pred_real)

        mse_scores.append(mse)
        mae_scores.append(mae)
        mape_scores.append(mape)

    return (
        np.mean(mse_scores), np.std(mse_scores),
        np.mean(mae_scores), np.std(mae_scores),
        np.mean(mape_scores), np.std(mape_scores)
    )


# ------------------ Regularisation BP-F ------------------

def regularization_experiments_bp_f():
    print("\n=== BP-F regularisation experiments ===")

    X_train, X_test, y_train, y_test, prep = train_test_split_bike()
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    alphas = [0.0001, 0.001, 0.01, 0.1]
    es_opts = [False, True]

    records = []

    for a in alphas:
        for es in es_opts:
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="tanh",
                learning_rate_init=0.01,
                max_iter=500,
                alpha=a,
                early_stopping=es,
                validation_fraction=0.2,
                random_state=RANDOM_STATE,
            )
            mlp.fit(X_train, y_train_scaled)
            y_pred = scaler_y.inverse_transform(
                mlp.predict(X_test).reshape(-1, 1)
            ).ravel()

            mse, mae, mape = regression_metrics(y_test, y_pred)
            records.append((a, es, mse, mae, mape))

    df = pd.DataFrame(records, columns=["alpha", "early_stopping", "MSE", "MAE", "MAPE"])
    print(df)
    return df


# ------------------ Main experiment runner ------------------

def run_all_models():
    X_train, X_test, y_train, y_test, preprocessor = train_test_split_bike()

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # ===== 1) MLR BASELINE =====
    mlr = LinearRegression()
    mlr.fit(X_train, y_train_scaled)
    y_pred_mlr = scaler_y.inverse_transform(
        mlr.predict(X_test).reshape(-1, 1)
    ).ravel()

    mse_mlr, mae_mlr, mape_mlr = regression_metrics(y_test, y_pred_mlr)
    print("=== MLR-F (baseline) ===")
    print(f"MSE: {mse_mlr:.3f}  MAE: {mae_mlr:.3f}  MAPE: {mape_mlr:.2f}%")

    # ===== 2) BP CONFIGS =====
    input_dim = X_train.shape[1]
    bp_configs = [
        {"layers": [input_dim, 16, 1], "epochs": 200, "lr": 0.01, "mom": 0.0, "act": "tanh"},
        {"layers": [input_dim, 32, 16, 1], "epochs": 300, "lr": 0.01, "mom": 0.5, "act": "tanh"},
        {"layers": [input_dim, 64, 32, 1], "epochs": 400, "lr": 0.005, "mom": 0.7, "act": "tanh"},
        {"layers": [input_dim, 32, 1], "epochs": 150, "lr": 0.02, "mom": 0.3, "act": "relu"},
    ]

    results = []
    for cfg in bp_configs:
        nn = NeuralNet(
            layers=cfg["layers"],
            n_epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            momentum=cfg["mom"],
            activation=cfg["act"],
            val_ratio=0.2,
            random_state=RANDOM_STATE,
        )
        nn.fit(X_train, y_train_scaled)
        y_pred_bp = scaler_y.inverse_transform(
            nn.predict(X_test).reshape(-1, 1)
        ).ravel()

        mse, mae, mape = regression_metrics(y_test, y_pred_bp)
        results.append({
            "layers": cfg["layers"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "mom": cfg["mom"],
            "act": cfg["act"],
            "val": 0.2,
            "mse": mse,
            "mae": mae,
            "mape": mape
        })

    print("\n=== BP (our implementation) – configs ===")
    print(pd.DataFrame(results))

    # ===== 3) BP-F (MLPRegressor) =====
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        learning_rate_init=0.01,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train_scaled)
    y_pred_mlp = scaler_y.inverse_transform(
        mlp.predict(X_test).reshape(-1, 1)
    ).ravel()

    mse_mlp, mae_mlp, mape_mlp = regression_metrics(y_test, y_pred_mlp)
    print("\n=== BP-F (MLPRegressor) ===")
    print(f"MSE: {mse_mlp:.3f}  MAE: {mae_mlp:.3f}  MAPE: {mape_mlp:.2f}%")

    # ===== 4) ENSEMBLES =====
    rf = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf, mae_rf, mape_rf = regression_metrics(y_test, y_pred_rf)

    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
    )
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    mse_gbr, mae_gbr, mape_gbr = regression_metrics(y_test, y_pred_gbr)

    print("\n=== Ensembles (optional) ===")
    print(f"Random Forest   -> MSE: {mse_rf:.3f}  MAE: {mae_rf:.3f}  MAPE: {mape_rf:.2f}%")
    print(f"Gradient Boost. -> MSE: {mse_gbr:.3f}  MAE: {mae_gbr:.3f}  MAPE: {mape_gbr:.2f}%")

    # ===== 5) REGULARISATION EXPERIMENTS =====
    regularization_experiments_bp_f()


if __name__ == "__main__":
    run_all_models()
