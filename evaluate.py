import argparse
import random
import statistics

import numpy
from pandas import DataFrame, read_csv
from scipy.io import loadmat

from logistic import LogisticRegressor, logistic_loss, quantize
from normalize import standardize, min_max_scale
from util import get_iris_sample, get_monk_sample
from visualize import decision_boundary


def test(data, model, sample_getter=get_iris_sample):
    """ Test a model on a given dataset. Return the accuracy an the loss over the provided dataset.

    :param data:                dataset
    :param model:               logistic regression model
    :return:        (tuple)     accuracy and loss
    """
    correct = 0
    loss = 0

    for i in range(len(data)):
        features, target = sample_getter(data, i)
        prediction = model.forward(features)
        sample_loss = logistic_loss(prediction, target)
        loss += sample_loss

        if quantize(prediction) == target:
            correct += 1

    return correct / len(data), loss / len(data)


def train(data, model, epochs=10, sample_getter=get_iris_sample):
    indices = list(range(len(data)))
    loss_trace = []
    averaged_trace = []
    epoch_trace = []
    for epoch in range(epochs):
        random.shuffle(indices)

        for i in indices:
            features, target = sample_getter(data, i)
            prediction = model.forward(features)
            model.backward(features, prediction, target)
            loss_trace.append(logistic_loss(prediction, target))

            if epoch > 0 or i >= 20:
                averaged_trace.append(statistics.mean(loss_trace[-20:]))

        epoch_trace.append(statistics.mean(loss_trace[-len(data):]))

    return epoch_trace


if __name__ == "__main__":
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["iris", "monk"], default="iris")
    parser.add_argument("--features", nargs="+", type=int, choices=[0, 1, 2, 3], default=[2, 3])
    args = parser.parse_args()

    safe = True

    # seeding
    numpy.random.seed(10)
    random.seed(10)

    # load data
    df = None
    if args.dataset == "iris":
        df: DataFrame = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                                 header=None)
        sample_getter = get_iris_sample
    elif args.dataset == "monk":
        df: DataFrame = DataFrame(loadmat("monk2.mat")["monk2"])
        sample_getter = get_monk_sample

    # features
    kept_features = args.features
    if args.dataset == "monk":
        kept_features = [0, 1, 2, 3, 4, 5]

    # select and split data
    data = df.drop(columns=[f for f in list(range(df.shape[1] - 1)) if f not in kept_features])
    data = data[:100] if args.dataset == "iris" else data
    data = standardize(data)
    data = data.sample(frac=1, random_state=10)

    train_set = data[:int(.8 * len(data))]
    test_set = data[int(.8 * len(data)):len(data)]

    # create model
    aggressor = LogisticRegressor(len(kept_features), weight_decay=0.05)

    train(train_set, aggressor, epochs=30, sample_getter=sample_getter)
    test_accuracy, test_loss = test(test_set, aggressor, sample_getter=sample_getter)
    train_accuracy, train_loss = test(train_set, aggressor, sample_getter=sample_getter)
    print(f"Accuracy: {test_accuracy} (test); {train_accuracy} (train).\n"
          f"Loss: {round(test_loss, 12)} (test); {round(train_loss, 12)} (train).")

    # visualize model
    if len(kept_features) == 2:
        decision_boundary(train_set, aggressor, (test_accuracy, test_loss, kept_features), safe=safe)
