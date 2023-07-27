import pickle
from abc import ABC

import numpy as np


class Module(ABC):
    """Abstract base class for all modules."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        """Save state"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """Load state"""
        with open(path, 'rb') as f:
            self = pickle.load(f)


class Parameter(Module):
    """A class for trainable model parameters."""
    pass
