from abc import ABC, abstractmethod
from .modules import Module
import numpy as np

"""https://www.v7labs.com/blog/neural-networks-activation-functions"""


class ActivationFunction(Module):
    @abstractmethod
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return "ActivationFunction()"


class Binary(ActivationFunction):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.where(x > self.threshold, 1, 0)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return np.zeros_like(dvalues)

    def __repr__(self):
        return "Binary()"


def binary(x: np.ndarray, threshold=0.0) -> np.ndarray:
    return np.where(x > threshold, 1, 0)


class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues

    def __repr__(self):
        return "Linear()"


def linear(x: np.ndarray) -> np.ndarray:
    return x


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues * self.outputs * (1 - self.outputs)

    def __repr__(self):
        return "Sigmoid()"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.tanh(x)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues * (1 - self.outputs**2)

    def __repr__(self):
        return "Tanh()"


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.where(x > 0, x, 0)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues * (self.outputs > 0)

    def __repr__(self):
        return "ReLU()"


def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


class LeakyReLU(ActivationFunction):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.where(x > 0, x, self.negative_slope * x)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues * np.where(self.outputs > 0, 1, self.negative_slope)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.negative_slope)

    def __repr__(self):
        return "LeakyReLU()"


def leaky_relu(x: np.ndarray, negative_slope=0.01) -> np.ndarray:
    return np.where(x > 0, x, negative_slope * x)


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self(x) + self.alpha)

    def __repr__(self):
        return "ELU()"


def elu(x: np.ndarray, alpha=1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


class Swish(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * sigmoid(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x) * (1 - x * (1 - sigmoid(x)))

    def __repr__(self):
        return "Swish()"


def swish(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


class GELU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (self(x) + x * (1 - tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)) ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))

    def __repr__(self):
        return "GELU()"


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Softmax:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=self.dim, keepdims=True)

    def __repr__(self):
        return "Softmax(dim={})".format(self.dim)


def softmax(x: np.ndarray, dim=1) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
