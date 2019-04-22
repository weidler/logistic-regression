import math

import numpy as np


class LogisticRegressor:

    def __init__(self, n_features, learning_rate=0.05, weight_decay=0):
        self.w = np.random.randn(n_features + 1)
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def forward(self, instance: np.ndarray):
        # prepend bias
        instance = self._augment_bias(instance)

        weighted = np.sum(self.w * instance)
        sigmoidal = sigmoid(weighted)

        return sigmoidal

    def backward(self, instance, prediction, target):
        instance = self._augment_bias(instance)
        for i in range(self.w.shape[0]):
            regularization = self.weight_decay * self.w[i] if i != 0 else 0
            gradient = (target - prediction) * instance[i] + regularization
            self.w[i] -= self.lr * -gradient

    @staticmethod
    def _augment_bias(instance):
        return np.insert(instance, 0, 1, axis=0)


def quantize(value):
    return 1 if value > 0.5 else 0


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.00000000000000001


def logistic_loss(prediction, target):
    prediction = 0.99999 if prediction == 1 else prediction
    return -target * math.log(prediction) - (1 - target) * math.log(1 - prediction)
