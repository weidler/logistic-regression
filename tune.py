import random

import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame, read_csv
from scipy.signal import savgol_filter

from evaluate import train
from logistic import LogisticRegressor
from normalize import standardize

if __name__ == "__main__":
    # seeding
    numpy.random.seed(100)
    random.seed(100)

    epochs = 100

    lr_scope = (0.05, 0.2)
    lr_step_size = 0.05
    lrs = [0.005] + [round(lr_scope[0] + step * lr_step_size, 3) for step in
           range(int((lr_scope[1] - lr_scope[0] + lr_step_size) // lr_step_size))]

    df: DataFrame = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    kept_features = [0, 2]
    data = df.drop(columns=[f for f in list(range(4)) if f not in kept_features])
    data = data[:100]
    data = data.sample(frac=1)
    data = standardize(data)

    loss_traces = []
    X = list(range(epochs))
    for lr in lrs:
        aggressor = LogisticRegressor(len(kept_features), learning_rate=lr, weight_decay=0)
        loss_traces.append(train(data, data[-2:-1], aggressor, epochs)[0])
        plt.plot(X, loss_traces[-1], label=f"{lr}")

    plt.legend()
    plt.show()
