# NeuralNet.py
# Custom feed-forward neural network with back-propagation
# for NEC Assignment 1.

import numpy as np


class NeuralNet:
    """
    Feed-forward neural network with back-propagation implemented from scratch.

    Variables (following NEC notation):
      - L: number of layers
      - n: number of units in each layer (including input and output)
      - h: list of field vectors h[l]
      - xi: list of activation vectors xi[l]
      - w: list of weight matrices w[l] (w[0] unused)
      - theta: list of threshold/bias vectors theta[l]
      - delta: list of backpropagated error vectors delta[l]
      - d_w, d_theta: current changes for weights and thresholds
      - d_w_prev, d_theta_prev: previous changes (momentum)
      - fact: activation function name ('sigmoid', 'tanh', 'relu', 'linear')
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
        """
        Parameters
        ----------
        layers : list[int]
            Architecture of the net. Example: [36, 64, 32, 1].
        n_epochs : int
            Number of training epochs.
        learning_rate : float
            Gradient descent learning rate.
        momentum : float
            Momentum coefficient in [0, 1).
        activation : str
            'sigmoid', 'tanh', 'relu' or 'linear'.
        val_ratio : float
            Fraction of patterns used for validation inside fit().
        random_state : int or None
            Seed for reproducible weight initialisation.
        """
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

        # Weights: w[l] connects layer (l-1) -> l ; w[0] unused
        self.w = [np.zeros((1, 1), dtype=float)]
        for l in range(1, self.L):
            # Xavier / Glorot initialisation
            limit = np.sqrt(6.0 / (self.n[l] + self.n[l - 1]))
            w_l = self.rng.uniform(-limit, limit, size=(self.n[l], self.n[l - 1]))
            self.w.append(w_l)

        # Updates for momentum
        self.d_w = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta = [np.zeros_like(th_l) for th_l in self.theta]
        self.d_w_prev = [np.zeros_like(w_l) for w_l in self.w]
        self.d_theta_prev = [np.zeros_like(th_l) for th_l in self.theta]

        # Loss history
        self._train_loss = []
        self._val_loss = []

    # ---------- activation functions ----------

    def _act(self, x):
        if self.fact == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.fact == "tanh":
            return np.tanh(x)
        if self.fact == "relu":
            return np.maximum(0.0, x)
        if self.fact == "linear":
            return x
        raise ValueError(f"Unknown activation function '{self.fact}'")

    def _act_deriv(self, x):
        if self.fact == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        if self.fact == "tanh":
            t = np.tanh(x)
            return 1.0 - t ** 2
        if self.fact == "relu":
            return (x > 0.0).astype(float)
        if self.fact == "linear":
            return np.ones_like(x)
        raise ValueError(f"Unknown activation function '{self.fact}'")

    # ---------- forward & backward ----------

    def _forward(self, x):
        """
        Feed-forward for a single pattern x (1D array of length n[0]).
        Stores h and xi for backprop and returns output.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n[0]:
            raise ValueError(
                f"Input dimension {x.shape[0]} does not match network "
                f"input size {self.n[0]}"
            )

        self.xi[0] = x

        for l in range(1, self.L):
            # h[l] = w[l] * xi[l-1] - theta[l]
            self.h[l] = self.w[l] @ self.xi[l - 1] - self.theta[l]
            self.xi[l] = self._act(self.h[l])

        return self.xi[-1]

    def _backward(self, y_target):
        """
        Backpropagation for one target y_target.
        Assumes _forward has just been called.
        """
        y_vec = np.atleast_1d(y_target).astype(float)
        if y_vec.shape[0] != self.n[-1]:
            raise ValueError(
                f"Target dimension {y_vec.shape[0]} does not match network "
                f"output size {self.n[-1]}"
            )

        Lm1 = self.L - 1

        # Output layer error: xi[L] - y
        error = self.xi[Lm1] - y_vec
        self.delta[Lm1] = error * self._act_deriv(self.h[Lm1])

        # Hidden layers
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = self._act_deriv(self.h[l]) * (self.w[l + 1].T @ self.delta[l + 1])

    # ---------- public API ----------

    def fit(self, X, y):
        """
        Online back-propagation with momentum.

        X: (n_samples, n_features)
        y: (n_samples,) or (n_samples, n_outputs)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")

        n_samples, n_features = X.shape
        if n_features != self.n[0]:
            raise ValueError(
                f"X has {n_features} features but network expects {self.n[0]}"
            )

        y = y.reshape(n_samples, self.n[-1])

        if self.val_ratio > 0.0:
            n_train = int((1.0 - self.val_ratio) * n_samples)
        else:
            n_train = n_samples

        indices = np.arange(n_samples)

        self._train_loss = []
        self._val_loss = []

        for _epoch in range(self.n_epochs):
            self.rng.shuffle(indices)
            X_shuf = X[indices]
            y_shuf = y[indices]

            X_train = X_shuf[:n_train]
            y_train = y_shuf[:n_train]
            X_val = X_shuf[n_train:] if n_train < n_samples else None
            y_val = y_shuf[n_train:] if n_train < n_samples else None

            sum_sq_error = 0.0

            for x_i, y_i in zip(X_train, y_train):
                y_pred = self._forward(x_i)
                self._backward(y_i)

                diff = y_pred - y_i
                sum_sq_error += np.dot(diff, diff)

                for l in range(1, self.L):
                    grad_w = np.outer(self.delta[l], self.xi[l - 1])
                    grad_theta = -self.delta[l]

                    self.d_w[l] = -self.lr * grad_w + self.momentum * self.d_w_prev[l]
                    self.d_theta[l] = -self.lr * grad_theta + self.momentum * self.d_theta_prev[l]

                    self.w[l] += self.d_w[l]
                    self.theta[l] += self.d_theta[l]

                    self.d_w_prev[l] = self.d_w[l]
                    self.d_theta_prev[l] = self.d_theta[l]

            train_mse = sum_sq_error / (len(X_train) * self.n[-1])
            self._train_loss.append(train_mse)

            if X_val is not None and len(X_val) > 0:
                sum_sq_error_val = 0.0
                for x_i, y_i in zip(X_val, y_val):
                    y_pred = self._forward(x_i)
                    diff = y_pred - y_i
                    sum_sq_error_val += np.dot(diff, diff)
                val_mse = sum_sq_error_val / (len(X_val) * self.n[-1])
            else:
                val_mse = None

            self._val_loss.append(val_mse)

    def predict(self, X):
        """
        Predict outputs for a batch of patterns.
        Returns 1D array if the output layer has a single unit.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        if n_features != self.n[0]:
            raise ValueError(
                f"X has {n_features} features but network expects {self.n[0]}"
            )

        outputs = np.zeros((n_samples, self.n[-1]), dtype=float)
        for i in range(n_samples):
            outputs[i] = self._forward(X[i])

        if self.n[-1] == 1:
            return outputs.ravel()
        return outputs

    def loss_epochs(self):
        """
        Returns (train_loss, val_loss) as NumPy arrays.
        """
        return np.array(self._train_loss, dtype=float), np.array(self._val_loss, dtype=float)
