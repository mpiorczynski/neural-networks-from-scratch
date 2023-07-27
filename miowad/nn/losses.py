from abc import ABC, abstractmethod

import numpy as np

from .activations import softmax


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, targets: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class MSELoss(LossFunction):
    def __call__(self, targets: np.ndarray, inputs: np.ndarray, reduction='mean') -> np.ndarray:
        self.inputs = inputs
        self.targets = targets

        if reduction == 'mean':
            return np.mean((self.targets - self.inputs) ** 2)
        elif reduction == 'sum':
            return np.sum((self.targets - self.inputs) ** 2)
        else:
            raise NotImplementedError(f'Unknown reduction {reduction}')

    def backward(self) -> np.ndarray:
        return 2 * (self.inputs - self.targets)


class CrossEntropyLoss(LossFunction):
    """LogSoftmax and NLLLoss"""

    def __call__(self, probabilities: np.ndarray, logits: np.ndarray, reduction='mean') -> np.ndarray:
        self.logits = logits
        self.probs = probabilities

        self.logits_softmax = softmax(self.logits)

        if reduction == 'mean':
            return np.mean(-np.sum(self.probs * np.log(self.logits_softmax), axis=1), axis=0)
        elif reduction == 'sum':
            return -np.sum(self.probs * np.log(self.logits_softmax))
        else:
            raise NotImplementedError(f'Unknown reduction {reduction}')

    def backward(self) -> np.ndarray:
        return self.logits_softmax - self.probs
