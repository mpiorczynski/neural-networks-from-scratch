import numpy as np

from .modules import Parameter
from miowad import nn
class Identity(Parameter):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def __repr__(self):
        return 'Identity()'


class Linear(Parameter):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(self.in_features, self.out_features)
        self.biases = np.zeros((self.out_features,))
        self.inputs = np.zeros((self.in_features,))
        self.dweights = np.zeros((self.in_features, self.out_features))
        self.dbiases = np.zeros((self.out_features,))
        self.init_method = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dweights = np.einsum('bi, bj->bij', self.inputs, dvalues).mean(axis=0)
        self.dbiases = np.mean(dvalues, axis=0)
        dvalues = np.dot(dvalues, self.weights.T)
        return dvalues

    def init_weights(self, init_method: str) -> None:
        self.init_method = init_method
        init_method_fn = getattr(nn.initializers, self.init_method)
        self.weights = init_method_fn((self.in_features, self.out_features))

    def __repr__(self):
        return 'Linear(in_features={}, out_features={})'.format(self.in_features, self.out_features)
