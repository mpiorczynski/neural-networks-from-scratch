import time

import matplotlib.pyplot as plt
import numpy as np

from .losses import LossFunction
from .modules import Module
from .optimizers import Optimizer
from ..utils import shuffle


class Trainer:
    def __init__(self, model: Module, optimizer: Optimizer, criterion: LossFunction, early_stopping=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopping = early_stopping
        if self.early_stopping is not None:
            self.early_stopping.set_trainer(self)
        self.history = {'train_loss': [], 'valid_loss': []}
        self.current_epoch = 0
        self.callbacks_metrics = {}

    def train(self, X_train, y_train, num_epochs, batch_size, X_valid=None, y_valid=None, log_every=100) -> None:
        assert len(X_train) == len(y_train)

        start = time.time()

        for epoch in range(num_epochs):
            self.current_epoch += 1

            # shuffle data
            X_train, y_train = shuffle(X_train, y_train)

            train_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                # create mini batches
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # forward pass
                outputs = self.model.forward(X_batch)

                # calculate loss
                train_loss += self.criterion(y_batch, outputs, reduction='sum')

                # backward pass
                dvalues = self.criterion.backward()
                self.model.backward(dvalues)

                # update weights and biases
                self.optimizer.step()

            train_loss /= len(X_train)
            self.history['train_loss'].append(train_loss)

            if epoch % log_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"train loss: {train_loss}")

            if X_valid is not None and y_valid is not None:
                outputs = self.model.forward(X_valid)
                valid_loss = self.criterion(y_valid, outputs)
                self.history['valid_loss'].append(valid_loss)
                if epoch % log_every == 0:
                    print(f"valid loss: {valid_loss}")

                if self.early_stopping is not None and self.early_stopping.run_early_stopping_check():
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        time_elapsed = time.time() - start
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def plot_training(self, xlog=False, ylog=False) -> None:
        plt.figure(figsize=(8, 6))
        plt.title("Training Loss")
        plt.plot(self.history['train_loss'], label="train")
        if 'valid_loss' in self.history:
            plt.plot(self.history['valid_loss'], label="valid")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')

        plt.legend(loc='best')
        plt.show()


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf
        self.monitor = None
        self.trainer = None

    def set_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def run_early_stopping_check(self) -> bool:
        if self.trainer.history["valid_loss"][-1] < self.best_score:
            self.best_score = self.trainer.history["valid_loss"][-1]
            self.counter = 0
            return False
        elif self.counter >= self.patience:
            return True
        else:
            self.counter += 1
            return False
