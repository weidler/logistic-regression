import argparse
import random
import statistics

import numpy
from pandas import DataFrame, read_csv
from scipy.io import loadmat

from logistic import LogisticRegressor, logistic_loss, quantize
from normalize import standardize
from util import get_iris_sample, get_monk_sample
from visualize import decision_boundary, explorative_data_analysis, performance_plot


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


def train(data, validation_data, model, epochs=10, sample_getter=get_iris_sample, involve_reg=False):
    indices = list(range(len(data)))
    train_loss_trace = []
    train_epoch_trace = []
    train_acc_trace = []
    test_epoch_trace = []
    test_acc_trace = []
    for epoch in range(epochs):
        random.shuffle(indices)

        epoch_correct = 0

        for i in indices:
            features, target = sample_getter(data, i)
            prediction = model.forward(features)
            model.backward(features, prediction, target)
            loss = logistic_loss(prediction, target)
            train_loss_trace.append(loss)
            if quantize(prediction) == target:
                epoch_correct += 1

        validation_acc, validation_loss = test(validation_data, model, sample_getter=sample_getter)
        test_epoch_trace.append(validation_loss)
        test_acc_trace.append(validation_acc)
        train_epoch_trace.append(statistics.mean(train_loss_trace[-len(data):]))
        train_acc_trace.append(epoch_correct / float(len(data)))

    return train_epoch_trace, test_epoch_trace, train_acc_trace, test_acc_trace


if __name__ == "__main__":
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["iris", "monk"], default="monk")
    parser.add_argument("--features", nargs="+", type=int, choices=[0, 1, 2, 3], default=[2, 3])
    parser.add_argument("--exploration", action="store_true", default=False)
    parser.add_argument("--performance", action="store_true", default=True)
    parser.add_argument("--decision-boundary", action="store_true", default=False)
    parser.add_argument("--no-plot", action="store_true", default=False)
    parser.add_argument("--safe", action="store_true", default=False)
    args = parser.parse_args()

    # seeding
    numpy.random.seed(100)
    random.seed(100)

    # load data
    df = None
    if args.dataset == "iris":
        df: DataFrame = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                                 header=None)
        sample_getter = get_iris_sample
    elif args.dataset == "monk":
        df: DataFrame = DataFrame(loadmat("data/monk2.mat")["monk2"])
        sample_getter = get_monk_sample

    # explore data
    if args.dataset == "iris" and args.exploration:
        explorative_data_analysis(df)

    # features
    kept_features = args.features
    if args.dataset == "monk":
        kept_features = [0, 1, 2, 3, 4, 5]

    # select and split data
    data = df.drop(columns=[f for f in list(range(df.shape[1] - 1)) if f not in kept_features])
    data = data[:100] if args.dataset == "iris" else data
    data = standardize(data)
    data = data.sample(frac=1, random_state=11)

    train_set = data[:int(.8 * len(data))]
    test_set = data[int(.8 * len(data)):len(data)]

    # create model
    aggressor = LogisticRegressor(len(kept_features), learning_rate=0.005, weight_decay=0)

    train_loss_trace, validation_loss_trace, train_acc_trace, validation_acc_trace = train(train_set, test_set,
                                                                                           aggressor, epochs=200,
                                                                                           sample_getter=sample_getter)
    test_accuracy, test_loss = test(test_set, aggressor, sample_getter=sample_getter)
    train_accuracy, train_loss = test(train_set, aggressor, sample_getter=sample_getter)
    print(f"Accuracy: {test_accuracy} (test); {train_accuracy} (train).\n"
          f"Loss: {round(test_loss, 12)} (test); {round(train_loss, 12)} (train).")

    if args.performance:
        performance_plot(train_loss_trace, validation_loss_trace, train_acc_trace, validation_acc_trace)

    # visualize model
    if len(kept_features) == 2 and args.decision_boundary:
        decision_boundary(test_set, aggressor, (train_accuracy, train_loss, kept_features), safe=args.safe,
                          plot=(not args.no_plot))
