"""
NeuralNet.py
------------
Custom implementation of a feed-forward neural network trained using
back-propagation with optional momentum.

This class was developed specifically for NEC Assignment 1 and serves as
a transparent implementation of the algorithm explained in the lectures.
It does not depend on external deep learning libraries.

Author: Nazim Alperen Akcakaya
"""

import numpy as np


class NeuralNet:
    """
    A fully-connected feed-forward neural network with:
        • Xavier weight initialization
        • Sigmoid / Tanh / ReLU activations
        • Online back-propagation
        • Optional momentum
        • Internal validation split (val_ratio)

    Parameters:
        layers        – list specifying neurons per layer (incl. input & output)
        n_epochs      – number of training epochs
        learning_rate – gradient descent step size
        momentum      – momentum coefficient
        activation    – activation function ('sigmoid', 'tanh', 'relu')
        val_ratio     – fraction of samples used for validation
        random_state  – RNG seed for reproducibility
    """

    def __init__(
        self,
        layers,
        n_epochs=100,
        learning_rate=0.01,
        momentum=0.0,
        activation="sigmoid",
        val_ratio=0.2,
        random_state=None,
    ):
        # Store architecture
        self.L = len(layers)        # Number of layers (input + hidden + output)
        self.n = list(layers)       # Neurons per layer

        # Training parameters
        self.n_epochs = int(n_epochs)
        self.lr = float(learning_rate)
        self.momentum = float(momentum)
        self.fact = activation.lower()
        self.val_ratio = float(val_ratio)

        # Random generator (used for weight init and shuffling)
        self.rng = np.random.default_rng(random_state)

        # Allocate space for internal variables
        self.h = [np.zeros(n_l) for n_l in self.n]       # Local field values
        self.xi = [np.zeros(n_l) for n_l in self.n]      # Activations
        self.theta = [np.zeros(n_l) for n_l in self.n]   # Biases
        self.delta = [np.zeros(n_l) for n_l in self.n]   # Backprop errors

        # -------------------------------
        # Xavier Initialization of weights
        # -------------------------------
        self.w = [np.zeros((1, 1))]
        for l in range(1, self.L):
            limit = np.sqrt(6.0 / (self.n[l] + self.n[l - 1]))
            w_l = self.rng.uniform(-limit, limit, size=(self.n[l], self.n[l - 1]))
            self.w.append(w_l)

        # Momentum buffers (initially zeros)
        self.d_w = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta = [np.zeros_like(th) for th in self.theta]
        self.d_w_prev = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta_prev = [np.zeros_like(th) for th in self.theta]

        # To store training/validation losses per epoch
        self._train_loss = []
        self._val_loss = []

    # ============================================================
    #                   ACTIVATION FUNCTIONS
    # ============================================================

    def _act(self, x):
        """Apply chosen activation function."""
        if self.fact == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.fact == "tanh":
            return np.tanh(x)
        if self.fact == "relu":
            return np.maximum(0.0, x)
        return x  # linear (identity)

    def _act_deriv(self, x):
        """Derivative of the chosen activation function."""
        if self.fact == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        if self.fact == "tanh":
            t = np.tanh(x)
            return 1.0 - t**2
        if self.fact == "relu":
            return (x > 0).astype(float)
        return np.ones_like(x)

    # ============================================================
    #                       FORWARD PASS
    # ============================================================

    def _forward(self, x):
        """
        Compute forward propagation for a single input vector.
        """
        x = np.asarray(x, dtype=float)
        self.xi[0] = x  # input layer has no activation function

        for l in range(1, self.L):
            # Local field: w*x - bias
            self.h[l] = self.w[l] @ self.xi[l - 1] - self.theta[l]
            # Apply activation
            self.xi[l] = self._act(self.h[l])

        return self.xi[-1]

    # ============================================================
    #                      BACKWARD PASS
    # ============================================================

    def _backward(self, y_target):
        """
        Compute backward propagation of errors for a single target.
        """
        y_vec = np.atleast_1d(y_target).astype(float)
        Lm1 = self.L - 1

        # Output layer delta
        error = self.xi[Lm1] - y_vec
        self.delta[Lm1] = error * self._act_deriv(self.h[Lm1])

        # Backpropagate to hidden layers
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = (
                self._act_deriv(self.h[l]) *
                (self.w[l + 1].T @ self.delta[l + 1])
            )

    # ============================================================
    #                           TRAINING
    # ============================================================

    def fit(self, X, y):
        """
        Train using online (sample-by-sample) back-propagation and momentum.
        A portion of the data (val_ratio) is used internally for validation.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(X.shape[0], self.n[-1])

        n_samples = X.shape[0]
        n_train = int((1.0 - self.val_ratio) * n_samples)

        # Indices for shuffling
        indices = np.arange(n_samples)

        self._train_loss = []
        self._val_loss = []

        for epoch in range(self.n_epochs):

            # Shuffle data every epoch
            self.rng.shuffle(indices)
            Xs = X[indices]
            ys = y[indices]

            # Split into training and validation subsets
            X_train = Xs[:n_train]
            y_train = ys[:n_train]
            X_val = Xs[n_train:]
            y_val = ys[n_train:]

            # Accumulate squared error for this epoch
            sum_sq_error = 0

            # ----------------------------
            # Online update per data point
            # ----------------------------
            for x_i, y_i in zip(X_train, y_train):
                y_pred = self._forward(x_i)
                self._backward(y_i)

                diff = y_pred - y_i
                sum_sq_error += np.dot(diff, diff)

                # Gradient updates
                for l in range(1, self.L):
                    grad_w = np.outer(self.delta[l], self.xi[l - 1])
                    grad_th = -self.delta[l]

                    # Momentum update rule
                    self.d_w[l] = -self.lr * grad_w + self.momentum * self.d_w_prev[l]
                    self.d_theta[l] = -self.lr * grad_th + self.momentum * self.d_theta_prev[l]

                    # Apply updates
                    self.w[l] += self.d_w[l]
                    self.theta[l] += self.d_theta[l]

                    # Store previous updates for momentum
                    self.d_w_prev[l] = self.d_w[l]
                    self.d_theta_prev[l] = self.d_theta[l]

            # Training MSE for epoch
            train_mse = sum_sq_error / (len(X_train) * self.n[-1])
            self._train_loss.append(train_mse)

            # Validation MSE
            if len(X_val) > 0:
                val_err = 0
                for x_i, y_i in zip(X_val, y_val):
                    diff = self._forward(x_i) - y_i
                    val_err += np.dot(diff, diff)
                self._val_loss.append(val_err / (len(X_val) * self.n[-1]))
            else:
                self._val_loss.append(None)

    # ============================================================
    #                           PREDICTION
    # ============================================================

    def predict(self, X):
        """
        Predict output for a single sample or a matrix of samples.
        """
        X = np.asarray(X, dtype=float)

        # Single vector → use forward directly
        if X.ndim == 1:
            return self._forward(X)

        # Batch prediction
        return np.array([self._forward(x) for x in X]).ravel()

    # ============================================================
    #                       LOSS ACCESSORS
    # ============================================================

    def loss_epochs(self):
        """
        Return arrays containing training and validation loss histories.
        Useful for plotting learning curves.
        """
        return np.array(self._train_loss), np.array(self._val_loss)
