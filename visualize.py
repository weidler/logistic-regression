import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame
import seaborn

from util import class_id


def decision_boundary(data: DataFrame, model, stats, safe=False, plot=True):
    data_x = data.iloc[:, 0]
    data_y = data.iloc[:, 1]
    data_class = [class_id(c) for c in data.iloc[:, 2]]

    # plot decision boundary
    step_size = .01
    x_min, x_max = data_x.min() - 1, data_x.max() + 1
    y_min, y_max = data_y.min() - 1, data_y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, step_size),
                            numpy.arange(y_min, y_max, step_size))
    Z = numpy.array(list(map(model.forward, numpy.c_[xx.ravel(), yy.ravel()])))

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap="hot", levels=1)

    # plot data points
    plt.scatter(x=data_x, y=data_y, c=data_class, cmap="tab10")

    plt.title(f"Features {stats[2][0]} and {stats[2][1]}; Accuracy: {stats[0]}, Loss: {'%.2E' % stats[1]}")

    if safe:
        plt.savefig(f"figures/db_{stats[2][0]}_{stats[2][1]}_test.pdf", format="pdf")

    if plot:
        plt.show()


def explorative_data_analysis(data: DataFrame):
    data = data.copy(deep=True)
    data.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"]

    seaborn.pairplot(data, hue="Species", palette="tab10", markers=["o", "s", "D"])
    plt.show()


def performance_plot(train_trace, validation_trace, train_acc, validation_acc):
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(train_trace, label="Training Loss")
    axs[0].plot(validation_trace, label="Validation Loss")

    axs[1].plot(train_acc, label="Training Accuracy")
    axs[1].plot(validation_acc, label="Validation Accuracy")

    axs[0].legend()
    axs[1].legend()
    plt.show()
