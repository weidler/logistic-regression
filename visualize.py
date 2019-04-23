import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame

from util import class_id


def decision_boundary(data: DataFrame, model, stats, safe=False):
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
        plt.savefig(f"figures/db_{stats[2][0]}_{stats[2][1]}.pdf", format="pdf")

    # plt.show()
