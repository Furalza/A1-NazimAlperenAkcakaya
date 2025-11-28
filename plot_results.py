import numpy as np
import matplotlib.pyplot as plt
from data_and_models import train_test_split_bike, regression_metrics
from NeuralNet import NeuralNet
import os

# ==============================================
# 1. Create folder for saving images
# ==============================================
output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# 2. Load preprocessed data
# ==============================================
X_train, X_test, y_train, y_test, _ = train_test_split_bike()

# Scale target for neural network
from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# ==============================================
# 3. Train your best BP model
# ==============================================
nn = NeuralNet(
    layers=[X_train.shape[1], 32, 16, 1],
    n_epochs=300,
    learning_rate=0.01,
    momentum=0.5,
    activation="tanh",
    val_ratio=0.2,
    random_state=42
)

nn.fit(X_train, y_train_scaled)

# Predictions (inverse transform)
pred_scaled = nn.predict(X_test)
pred_real = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

# ==============================================
# 4. Scatter plot (Predicted vs True)
# ==============================================
plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred_real, s=10, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("True cnt")
plt.ylabel("Predicted cnt")
plt.title("Scatter Plot – True vs Predicted (BP)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_bp.png", dpi=300)
plt.close()

# ==============================================
# 5. Learning curves (Training vs Validation Loss)
# ==============================================
train_loss, val_loss = nn.loss_epochs()

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Learning Curve – BP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/learning_curve_bp.png", dpi=300)
plt.close()

print("All figures saved in:", output_dir)
