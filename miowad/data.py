import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Dataset(ABC):
    __slots__ = ['name', 'data_path', 'type', 'train_data', 'test_data', 'train_df', 'test_df']

    def __init__(self, name: str):
        self.name = name
        self.data_path = 'data'
        self.train_data = os.path.join(self.data_path, self.type, self.name + '-training.csv')
        self.test_data = os.path.join(self.data_path, self.type, self.name + '-test.csv')

    def to_df(self) -> (pd.DataFrame, pd.DataFrame):
        train_df = pd.read_csv(self.train_data)
        test_df = pd.read_csv(self.test_data)
        return train_df, test_df


class RegressionDataset(Dataset):
    def __init__(self, name):
        self.type = 'regression'
        super().__init__(name)

    def plot_dataset(self) -> None:
        df_train, df_test = self.to_df()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(data=df_train, x='x', y='y', color='blue', legend=False, ax=ax1, )
        ax1.set_title("Training data")
        sns.scatterplot(data=df_test, x='x', y='y', color='green', legend=False, ax=ax2, )
        ax2.set_title("Test data")
        plt.show()


class ClassificationDataset(Dataset):
    def __init__(self, name):
        self.type = 'classification'
        super().__init__(name)

    def plot_dataset(self) -> None:
        df_train, df_test = self.to_df()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(data=df_train, x='x', y='y', hue='c', legend=False, ax=ax1, )
        ax1.set_title("Training data")
        sns.scatterplot(data=df_test, x='x', y='y', hue='c', legend=False, ax=ax2, )
        ax2.set_title("Test data")
        plt.show()


class Transformer(ABC):
    def fit(self, x: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass


class MinMaxScaler(Transformer):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x: np.ndarray) -> None:
        self.min = x.min(axis=0, keepdims=True)
        self.max = x.max(axis=0, keepdims=True)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max - self.min) + self.min


class StandardScaler(Transformer):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class OneHotEncoder(Transformer):
    """Encode categorical features as a one-hot vector"""

    def __init__(self):
        self.n_classes = None

    def fit(self, x: np.ndarray) -> None:
        self.n_classes = len(np.unique(x, axis=0, ))

    def transform(self, x: np.ndarray) -> np.ndarray:
        encoded = np.zeros((x.shape[0], self.n_classes))
        encoded[np.arange(x.shape[0]), x.astype(int)] = 1
        return encoded

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(x, axis=1)
