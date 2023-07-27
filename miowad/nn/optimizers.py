import pickle
from abc import ABC
from typing import List

import numpy as np

from .modules import Parameter


"""https://www.ruder.io/optimizing-gradient-descent/"""


class Optimizer(ABC):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.dweights = np.zeros_like(param.dweights)
            param.dbiases = np.zeros_like(param.dbiases)

    def save_state(self, path: str) -> None:
        """Save state"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_state(self, path: str) -> None:
        """Load state"""
        with open(path, "rb") as f:
            self = pickle.load(f)


class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__(parameters, lr, weight_decay)

    def step(self) -> None:
        # update weights and biases
        for param in self.parameters:
            if self.weight_decay != 0:
                param.dweights += self.weight_decay * param.weights

            param.weights -= self.lr * param.dweights
            param.biases -= self.lr * param.dbiases


class Momentum(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [{"weights": np.zeros_like(param.weights), "biases": np.zeros_like(param.biases)} for param in self.parameters]

    def step(self) -> None:
        for idx, (v, param) in enumerate(zip(self.velocities, self.parameters)):
            if self.weight_decay != 0:
                param.dweights += self.weight_decay * param.weights

            # update velocity
            v["weights"] += self.momentum * param.dweights
            v["biases"] += self.momentum * param.dbiases

            # save velocity
            self.velocities[idx]["weights"] = v["weights"]
            self.velocities[idx]["biases"] = v["biases"]

            param.weights -= self.lr * v["weights"]
            param.biases -= self.lr * v["biases"]


class NAG(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, momentum: float = 0.01, weight_decay: float = 0.0):
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum


class RMSProp(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, alpha: float = 0.9, eps: float = 1e-08, weight_decay: float = 0.0):
        super().__init__(parameters, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.velocities = [{"weights": np.zeros_like(param.weights), "biases": np.zeros_like(param.biases)} for param in self.parameters]

    def step(self):
        for idx, (v, param) in enumerate(zip(self.velocities, self.parameters)):
            if self.weight_decay != 0:
                param.dweights += self.weight_decay * param.weights

            # update velocity
            v["weights"] = self.alpha * v["weights"] + (1 - self.alpha) * param.dweights**2
            v["biases"] = self.alpha * v["biases"] + (1 - self.alpha) * param.dbiases**2

            # save velocity
            self.velocities[idx]["weights"] = v["weights"]
            self.velocities[idx]["biases"] = v["biases"]

            # update param
            param.weights -= self.lr * param.dweights / np.sqrt(v["weights"] + self.eps)
            param.biases -= self.lr * param.dbiases / np.sqrt(v["biases"] + self.eps)


class Adam(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-08, weight_decay: float = 0.0):
        super().__init__(parameters, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1
        self.m = [{"weights": np.zeros_like(param.weights), "biases": np.zeros_like(param.biases)} for param in self.parameters]
        self.v = [{"weights": np.zeros_like(param.weights), "biases": np.zeros_like(param.biases)} for param in self.parameters]

    def step(self):
        for idx, (m, v, param) in enumerate(zip(self.m, self.v, self.parameters)):
            if self.weight_decay != 0:
                param.dweights += self.weight_decay * param.weights

            # first moment estimate
            m["weights"] = self.beta1 * m["weights"] + (1 - self.beta1) * param.dweights
            m["biases"] = self.beta1 * m["biases"] + (1 - self.beta1) * param.dbiases

            # second moment estimate
            v["weights"] = self.beta2 * v["weights"] + (1 - self.beta2) * param.dweights**2
            v["biases"] = self.beta2 * v["biases"] + (1 - self.beta2) * param.dbiases**2

            # bias correction
            m["weights"] /= 1 - self.beta1**self.t
            m["biases"] /= 1 - self.beta1**self.t

            v["weights"] /= 1 - self.beta2**self.t
            v["biases"] /= 1 - self.beta2**self.t

            # save moments
            self.m[idx]["weights"] = m["weights"]
            self.m[idx]["biases"] = m["biases"]

            self.v[idx]["weights"] = v["weights"]
            self.v[idx]["biases"] = v["biases"]

            # update param
            param.weights -= self.lr * m["weights"] / np.sqrt(v["weights"] + self.eps)
            param.biases -= self.lr * m["biases"] / np.sqrt(v["biases"] + self.eps)

        self.t += 1
