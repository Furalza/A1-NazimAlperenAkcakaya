# NeuralNet.py
# Base structure for custom neural network (NEC Assignment 1)
# Detailed implementation to be added in further commits.

import numpy as np

class NeuralNet:
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

        # Allocate main internal structures (empty for now)
        self.h = [np.zeros(n_l, dtype=float) for n_l in self.n]
        self.xi = [np.zeros(n_l, dtype=float) for n_l in self.n]
        self.theta = [np.zeros(n_l, dtype=float) for n_l in self.n]
        self.delta = [np.zeros(n_l, dtype=float) for n_l in self.n]

        # Weight matrices (initialised later)
        self.w = [np.zeros((1, 1), dtype=float) for _ in range(self.L)]

        # For momentum
        self.d_w = [None] * self.L
        self.d_theta = [None] * self.L
        self.d_w_prev = [None] * self.L
        self.d_theta_prev = [None] * self.L

        # Loss history
        self._train_loss = []
        self._val_loss = []

    # Methods will be implemented in later commits:
    # - _act()
    # - _act_deriv()
    # - _forward()
    # - _backward()
    # - fit()
    # - predict()
    # - loss_epochs()
