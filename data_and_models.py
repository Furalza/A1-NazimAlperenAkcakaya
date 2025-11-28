"""
data_and_models.py
------------------
Main experiment script for Activity 1 (NEC – URV).  
Contains:
    • Data loading and preprocessing  
    • Train/test split (80/20 with shuffle)
    • Custom BP model experiments  
    • BP-F (MLPRegressor)  
    • Ensemble models (optional)
    • Regularisation experiments (optional)
    • K-Fold cross-validation utilities (optional)

Author: Nazim Alperen Akcakaya
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from NeuralNet import NeuralNet

# Fixed seed for reproducibility
RANDOM_STATE = 42


# ============================================================
#                        METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Compute regression evaluation metrics:
        • MSE  – Mean Squared Error
        • MAE  – Mean Absolute Error
        • MAPE – Mean Absolute Percentage Error

    A small constant prevents division by zero in MAPE.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return mse, mae, mape


# ============================================================
#                   DATA LOADING & CLEANING
# ============================================================

def load_bike_data(path="data/hour.csv"):
    """
    Load the Bike Sharing Dataset and remove columns that leak the target or
    behave as identifiers.

    NOTE: For the assignment, hour.csv should be used because it contains
          more than 1000 patterns.

    Returns:
        X          – input feature DataFrame
        y          – target vector
        num_cols   – numerical columns
        cat_cols   – categorical columns
    """
    df = pd.read_csv(path)

    # Remove features that leak target or are identifiers
    drop_cols = ["instant", "dteday", "casual", "registered"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Extract target
    y = df["cnt"].astype(float).values
    X = df.drop(columns=["cnt"])

    # Define variable types
    cat_cols = ["season", "yr", "mnth", "holiday", "weekday",
                "workingday", "weathersit"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols


# ============================================================
#                  TRAIN / TEST SPLIT + PREPROCESSING
# ============================================================

def train_test_split_bike(test_size=0.2):
    """
    Split dataset into training+validation (80%) and test (20%), after shuffling.
    Apply scaling and one-hot encoding using ColumnTransformer.

    The preprocessor is fitted only on the training data to avoid leakage.
    """
    X, y, num_cols, cat_cols = load_bike_data()

    # Preprocessing pipeline for both numerical and categorical attributes
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Correct random split as required by the assignment
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Fit preprocessing only on training data
    X_train_proc = preprocessor.fit_transform(X_train_full)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train_full, y_test, preprocessor


# ============================================================
#              CUSTOM BP NEURAL NETWORK – K-FOLD CV
# ============================================================

def kfold_cv_neuralnet(X, y, layers, n_epochs, lr, mom, activation, k=5):
    """
    Perform K-fold cross-validation for the custom BP neural network.
    Inside each fold, the BP model internally splits its training data
    into train/validation according to val_ratio.

    Returns aggregated metrics across folds:
        mean MSE, std MSE, mean MAE, std MAE, mean MAPE, std MAPE
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    mse_scores, mae_scores, mape_scores = [], [], []

    # Scale target for neural network stability
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_s, y_test = y_scaled[train_idx], y[test_idx]

        # Train custom BP model
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

        # Predict in original target space
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


# ============================================================
#                REGULARISATION EXPERIMENTS (BP-F)
# ============================================================

def regularization_experiments_bp_f():
    """
    Test effect of L2 (alpha) regularisation and early stopping
    on the BP-F (MLPRegressor) model.

    Returns results as a DataFrame.
    """
    print("\n=== BP-F regularisation experiments ===")

    X_train, X_test, y_train, y_test, _ = train_test_split_bike()

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
                alpha=a,                 # L2 regularisation
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


# ============================================================
#                     MAIN EXPERIMENT PIPELINE
# ============================================================

def run_all_models():
    """
    Run all models defined in the assignment:
        1. MLR-F baseline
        2. Custom BP with several configurations
        3. BP-F (MLPRegressor)
        4. Ensemble models (Random Forest, Gradient Boosting)
        5. Regularisation experiments
    """
    X_train, X_test, y_train, y_test, _ = train_test_split_bike()

    # Scale target for neural network methods
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------
    # 1. MLR-F BASELINE
    # ------------------------------------------------------------
    mlr = LinearRegression()
    mlr.fit(X_train, y_train_scaled)

    y_pred_mlr = scaler_y.inverse_transform(
        mlr.predict(X_test).reshape(-1, 1)
    ).ravel()

    mse_mlr, mae_mlr, mape_mlr = regression_metrics(y_test, y_pred_mlr)
    print("=== MLR-F (baseline) ===")
    print(f"MSE: {mse_mlr:.3f}  MAE: {mae_mlr:.3f}  MAPE: {mape_mlr:.2f}%")

    # ------------------------------------------------------------
    # 2. CUSTOM BP CONFIGURATIONS
    # ------------------------------------------------------------
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

        # Back to original target scale
        y_pred_bp = scaler_y.inverse_transform(
            nn.predict(X_test).reshape(-1, 1)
        ).ravel()

        mse, mae, mape = regression_metrics(y_test, y_pred_bp)
        results.append({
            **cfg,
            "val": 0.2,
            "mse": mse,
            "mae": mae,
            "mape": mape
        })

    print("\n=== BP (our implementation) – configs ===")
    print(pd.DataFrame(results))

    # ------------------------------------------------------------
    # 3. BP-F MODEL (MLPRegressor)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 4. ENSEMBLE MODELS
    # ------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)

    mse_rf, mae_rf, mape_rf = regression_metrics(y_test, y_pred_rf)
    mse_gbr, mae_gbr, mape_gbr = regression_metrics(y_test, y_pred_gbr)

    print("\n=== Ensembles (optional) ===")
    print(f"Random Forest   -> MSE: {mse_rf:.3f}  MAE: {mae_rf:.3f}  MAPE: {mape_rf:.2f}%")
    print(f"Gradient Boost. -> MSE: {mse_gbr:.3f}  MAE: {mae_gbr:.3f}  MAPE: {mape_gbr:.2f}%")

    # ------------------------------------------------------------
    # 5. REGULARISATION EXPERIMENTS (BP-F)
    # ------------------------------------------------------------
    regularization_experiments_bp_f()


# ============================================================
#                        ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_all_models()
