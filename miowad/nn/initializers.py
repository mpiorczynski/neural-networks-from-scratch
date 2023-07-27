from typing import Tuple

import numpy as np


def uniform(shape: Tuple, a=0.0, b=1.0, scale=1.0) -> np.ndarray:
    return scale * np.random.uniform(a, b, shape)


def normal(shape: Tuple, mean=0.0, std=1.0, scale=1.0) -> np.ndarray:
    return scale * (mean + std * np.random.randn(*shape))


def xavier_uniform(shape: Tuple) -> np.ndarray:
    """Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)"""
    n_in, n_out = shape
    a = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-a, a, shape)


def xavier_normal(shape: Tuple) -> np.ndarray:
    """Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)"""
    n_in, n_out = shape
    std = np.sqrt(2.0 / (n_in + n_out))
    return std * np.random.randn(*shape)


def he_uniform(shape: Tuple) -> np.ndarray:
    """Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)"""
    n_in, _ = shape
    a = np.sqrt(6.0 / n_in)
    return np.random.uniform(-a, a, shape)


def he_normal(shape: Tuple) -> np.ndarray:
    """Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)"""
    n_in, _ = shape
    std = np.sqrt(2.0 / n_in)
    return std * np.random.randn(*shape)
