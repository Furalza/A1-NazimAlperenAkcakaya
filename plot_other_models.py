import numpy as np
import matplotlib.pyplot as plt
from data_and_models import train_test_split_bike
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)

# ===========================================================
# Load data
# ===========================================================
X_train, X_test, y_train, y_test, _ = train_test_split_bike()

# Target scaling for BP-F model
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# ===========================================================
# BP-F MODEL (MLPRegressor)
# ===========================================================
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="tanh",
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train_scaled)

y_pred_mlp_scaled = mlp.predict(X_test)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_mlp, s=10, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("True cnt")
plt.ylabel("Predicted cnt")
plt.title("Scatter Plot – True vs Predicted (BP-F / MLPRegressor)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_bpf.png", dpi=300)
plt.close()

# ===========================================================
# RANDOM FOREST
# ===========================================================
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_rf, s=10, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("True cnt")
plt.ylabel("Predicted cnt")
plt.title("Scatter Plot – True vs Predicted (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_rf.png", dpi=300)
plt.close()

print("Figures generated:")
print(" - scatter_bpf.png")
print(" - scatter_rf.png")
