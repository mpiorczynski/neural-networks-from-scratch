import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolut Error"""
    return np.mean(abs(y_true - y_pred))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Accuracy"""
    return np.mean(y_true == y_pred)
