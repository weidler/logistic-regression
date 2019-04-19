import math

import numpy as np
import pandas as pd


class LogisticRegressor:

    def __init__(self):
        self.w = np.random.randn(3)
        self.lr = 0.05

    def forward(self, instance: np.ndarray):
        # prepend bias
        instance = self._augment_bias(instance)

        weighted = np.sum(self.w * instance)
        logisted = sigmoid(weighted)

        return logisted

    def backward(self, instance, prediction, target):
        instance = self._augment_bias(instance)
        for i in range(self.w.shape[0]):
            gradient = (target - prediction) * instance[i]
            self.w[i] -= self.lr * -gradient

    @staticmethod
    def _augment_bias(instance):
        return np.insert(instance, 0, 1, axis=0)


def normalize(data: pd.DataFrame):
    """Normalize using z-score standardization"""
    means = data.mean()
    stds = data.std()

    for index in means.index:
        data[index] = data[index].apply(lambda x: ((x - means[index])/stds[index]))

    return data


def quantize(value):
    return 1 if value > 0.5 else 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logistic_loss(prediction, target):
    return -target * math.log(prediction) - (1 - target) * math.log(1 - prediction)


def get_sample(dataset: pd.DataFrame):
    row = dataset.sample().values[0]
    return row[:2], class_id(row[2])


def class_id(classname: str):
    if classname == "Iris-versicolor":
        return 0
    elif classname == "Iris-setosa":
        return 1
    else:
        raise ValueError("Unknown Class!")


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                                   header=None)

    kept_features = [0, 2]

    data = df.drop(columns=[f for f in list(range(4)) if f not in kept_features])
    data = data[:100]
    data = normalize(data)

    aggressor = LogisticRegressor()

    def test(data, repetitions=1000, verbose=False):
        total_loss = 0
        repetitions = 100
        for i in range(repetitions):
            features, target = get_sample(data)
            prediction = aggressor.forward(features)
            loss = logistic_loss(prediction, target)
            total_loss += loss

            if verbose:
                print(f"{prediction} | {target} | {loss}")

        print(f"TOTAL LOSS: {total_loss / repetitions}")


    test()
    # train
    for i in range(10000):
        features, target = get_sample(data)
        prediction = aggressor.forward(features)
        aggressor.backward(features, prediction, target)
    test()
