from pandas import DataFrame, read_csv

from evaluate import train
from logistic import LogisticRegressor
from normalize import standardize

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

if __name__ == "__main__":
    lr_scope = (0.01, 0.1)
    lr_step_size = 0.01
    lrs = [round(lr_scope[0] + step * lr_step_size, 3) for step in
           range(int((lr_scope[1] - lr_scope[0] + lr_step_size) // lr_step_size))]

    df: DataFrame = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    kept_features = [0, 2]
    data = df.drop(columns=[f for f in list(range(4)) if f not in kept_features])
    data = data[:100]
    data = standardize(data)

    loss_traces = []
    X = list(range(1, 100 * 5 + 1))
    for lr in lrs:
        print(lr)
        try:
            aggressor = LogisticRegressor(len(kept_features), learning_rate=lr, weight_decay=0.05)
            loss_traces.append(train(data, aggressor, 5))
            plt.plot(X, savgol_filter(loss_traces[-1], 101, 2), label=f"{lr}")
        except:
            pass

    plt.legend()
    plt.show()
