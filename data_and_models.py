# data_and_models.py
# Data loading and preprocessing skeleton for NEC Assignment 1.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from NeuralNet import NeuralNet

RANDOM_STATE = 42


def regression_metrics(y_true, y_pred):
    """Return (MSE, MAE, MAPE) for regression."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return mse, mae, mape


def load_bike_data(path="data/day.csv"):
    """
    Load Bike Sharing (daily) dataset from CSV.
    """
    df = pd.read_csv(path)

    # Drop columns that are identifiers or leak the target
    drop_cols = ["instant", "dteday", "casual", "registered"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Target
    y = df["cnt"].astype(float).values
    X = df.drop(columns=["cnt"])

    # Categorical vs numerical
    cat_cols = ["season", "yr", "mnth", "holiday", "weekday",
                "workingday", "weathersit"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols


def train_test_split_bike(test_size=0.2):
    """
    Train/test split + preprocessing pipeline.
    """
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


def run_all_models():
    """
    Placeholder – models will be added in later commits.
    """
    X_train, X_test, y_train, y_test, preprocessor = train_test_split_bike()
    print("Data shapes:", X_train.shape, X_test.shape)


if __name__ == "__main__":
    run_all_models()
