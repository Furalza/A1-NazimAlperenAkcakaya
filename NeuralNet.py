# NeuralNet.py
# Custom feed-forward neural network with back-propagation
# for NEC Assignment 1.

import numpy as np


class NeuralNet:
    """
    Feed-forward neural network with back-propagation implemented from scratch.
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
        # Architecture
        self.L = len(layers)
        self.n = list(layers)

        self.n_epochs = int(n_epochs)
        self.lr = float(learning_rate)
        self.momentum = float(momentum)
        self.fact = activation.lower()
        self.val_ratio = float(val_ratio)

        # Random generator
        self.rng = np.random.default_rng(random_state)

        # Fields and activations
        self.h = [np.zeros(n_l, dtype=float) for n_l in self.n]
        self.xi = [np.zeros(n_l, dtype=float) for n_l in self.n]

        # Thresholds / biases
        self.theta = [np.zeros(n_l, dtype=float) for n_l in self.n]

        # Error terms
        self.delta = [np.zeros(n_l, dtype=float) for n_l in self.n]

        # Weights (Xavier init)
        self.w = [np.zeros((1, 1), dtype=float)]
        for l in range(1, self.L):
            limit = np.sqrt(6.0 / (self.n[l] + self.n[l - 1]))
            w_l = self.rng.uniform(-limit, limit, size=(self.n[l], self.n[l - 1]))
            self.w.append(w_l)

        # Momentum buffers
        self.d_w = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta = [np.zeros_like(th_l) for th_l in self.theta]
        self.d_w_prev = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta_prev = [np.zeros_like(th_l) for th_l in self.theta]

        # Loss history
        self._train_loss = []
        self._val_loss = []

    # ------------------ Activation functions ------------------

    def _act(self, x):
        if self.fact == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.fact == "tanh":
            return np.tanh(x)
        if self.fact == "relu":
            return np.maximum(0.0, x)
        return x

    def _act_deriv(self, x):
        if self.fact == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        if self.fact == "tanh":
            t = np.tanh(x)
            return 1.0 - t**2
        if self.fact == "relu":
            return (x > 0.0).astype(float)
        return np.ones_like(x)

    # ------------------ Forward pass ------------------

    def _forward(self, x):
        x = np.asarray(x, dtype=float)
        self.xi[0] = x

        for l in range(1, self.L):
            self.h[l] = self.w[l] @ self.xi[l - 1] - self.theta[l]
            self.xi[l] = self._act(self.h[l])

        return self.xi[-1]

    # ------------------ Backward pass ------------------

    def _backward(self, y_target):
        y_vec = np.atleast_1d(y_target).astype(float)
        Lm1 = self.L - 1

        error = self.xi[Lm1] - y_vec
        self.delta[Lm1] = error * self._act_deriv(self.h[Lm1])

        for l in range(self.L - 2, 0, -1):
            self.delta[l] = self._act_deriv(self.h[l]) * (self.w[l + 1].T @ self.delta[l + 1])

    # ------------------ Fit (Online BP + Momentum) ------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(X.shape[0], self.n[-1])

        n_samples = X.shape[0]
        n_train = int((1.0 - self.val_ratio) * n_samples)
        indices = np.arange(n_samples)

        self._train_loss = []
        self._val_loss = []

        for epoch in range(self.n_epochs):
            self.rng.shuffle(indices)
            Xs = X[indices]
            ys = y[indices]

            X_train = Xs[:n_train]
            y_train = ys[:n_train]
            X_val = Xs[n_train:]
            y_val = ys[n_train:]

            sum_sq_error = 0

            for x_i, y_i in zip(X_train, y_train):
                y_pred = self._forward(x_i)
                self._backward(y_i)

                diff = y_pred - y_i
                sum_sq_error += np.dot(diff, diff)

                for l in range(1, self.L):
                    grad_w = np.outer(self.delta[l], self.xi[l - 1])
                    grad_th = -self.delta[l]

                    self.d_w[l] = -self.lr * grad_w + self.momentum * self.d_w_prev[l]
                    self.d_theta[l] = -self.lr * grad_th + self.momentum * self.d_theta_prev[l]

                    self.w[l] += self.d_w[l]
                    self.theta[l] += self.d_theta[l]

                    self.d_w_prev[l] = self.d_w[l]
                    self.d_theta_prev[l] = self.d_theta[l]

            train_mse = sum_sq_error / (len(X_train) * self.n[-1])
            self._train_loss.append(train_mse)

            if len(X_val) > 0:
                val_err = 0
                for x_i, y_i in zip(X_val, y_val):
                    y_pred = self._forward(x_i)
                    diff = y_pred - y_i
                    val_err += np.dot(diff, diff)
                self._val_loss.append(val_err / (len(X_val) * self.n[-1]))
            else:
                self._val_loss.append(None)

    # ------------------ Predict ------------------

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self._forward(X)
        return np.array([self._forward(x) for x in X]).ravel()

    # ------------------ Loss history ------------------

    def loss_epochs(self):
        return np.array(self._train_loss), np.array(self._val_loss)
