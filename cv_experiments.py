from data_and_models import (
    load_bike_data,
    kfold_cv_neuralnet,
    RANDOM_STATE,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X, y, num_cols, cat_cols = load_bike_data()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_all = preprocessor.fit_transform(X)

mean_mse, std_mse, mean_mae, std_mae, mean_mape, std_mape = \
    kfold_cv_neuralnet(
        X_all,
        y,
        layers=[X_all.shape[1], 64, 32, 1],
        n_epochs=200,
        lr=0.01,
        mom=0.5,
        activation="tanh",
        k=5,
    )

print("=== 5-fold CV for BP ===")
print(f"MSE:  {mean_mse:.3f} ± {std_mse:.3f}")
print(f"MAE:  {mean_mae:.3f} ± {std_mae:.3f}")
print(f"MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")
