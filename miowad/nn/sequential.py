from typing import List

import numpy as np

from .modules import Module, Parameter


class Sequential(Module):
    """A sequential container for neural network modules."""

    def __init__(self, layers: List[Module]) -> None:
        self.init_method = None
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            if isinstance(layer, Parameter):
                self.parameters.append(layer)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def add(self, layer: Module):
        self.layers.append(layer)
        if isinstance(layer, Parameter):
            self.parameters.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
        return dvalues

    def init_weights(self, init_method: str = 'normal') -> None:
        self.init_method = init_method
        for param in self.parameters:
            param.init_weights(init_method)

    def __len__(self):
        return len(self.parameters)

    def __repr__(self) -> str:
        s = "Sequential([\n"
        for layer in self.layers:
            s += "\t" + str(layer) + "\n"
        return s + "])"
