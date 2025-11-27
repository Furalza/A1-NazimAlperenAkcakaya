"""
Five-fold cross-validation experiment for the custom Back-Propagation (BP) model.
The full dataset is preprocessed using a ColumnTransformer, which applies a
StandardScaler to all numerical variables and one-hot encoding to categorical variables.
The transformed data is then evaluated using kfold_cv_neuralnet(), which implements an
internal train/validation split for each fold, following the specifications of the assignment.
"""

from data_and_models import load_bike_data, kfold_cv_neuralnet, RANDOM_STATE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset and identify variable types
X, y, num_cols, cat_cols = load_bike_data()

# Preprocessing pipeline: scaling numeric attributes and encoding categorical ones
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# Fit preprocessing on the full dataset and obtain the preprocessed matrix
X_all = preprocessor.fit_transform(X)

# Execute 5-fold cross-validation for the BP model
mean_mse, std_mse, mean_mae, std_mae, mean_mape, std_mape = kfold_cv_neuralnet(
    X_all,
    y,
    layers=[X_all.shape[1], 64, 32, 1],   # Architecture tested
    n_epochs=200,
    lr=0.01,
    mom=0.5,
    activation="tanh",
    k=5,
)

# Print the aggregated results across the folds
print("=== 5-fold CV for BP ===")
print(f"MSE:  {mean_mse:.3f} ± {std_mse:.3f}")
print(f"MAE:  {mean_mae:.3f} ± {std_mae:.3f}")
print(f"MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")
