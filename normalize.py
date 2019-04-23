from pandas import DataFrame


def standardize(data: DataFrame):
    """Normalize using z-score standardization"""
    means = data.iloc[:, :-1].mean()
    stds = data.iloc[:, :-1].std()

    for index in means.index:
        data[index] = data[index].apply(lambda x: ((x - means[index])/stds[index]))

    return data


def min_max_scale(data: DataFrame):
    """Normalize using min max scaling"""
    minimums = data.iloc[:, :-1].min()
    maximums = data.iloc[:, :-1].max()

    for index in minimums.index:
        data[index] = data[index].apply(lambda x: ((x - minimums[index]) / (maximums[index] - minimums[index])))

    return data
