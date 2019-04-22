from pandas import DataFrame


def class_id(classname: str):
    if isinstance(classname, int):
        return classname

    if classname == "Iris-versicolor":
        return 0
    elif classname == "Iris-setosa":
        return 1
    else:
        raise ValueError(f"Unknown Class {classname}!")


def get_iris_sample(data: DataFrame, index):
    row = data.iloc[[index]]
    return row.values[0][:-1], class_id(row.values[0][-1])


def get_monk_sample(data: DataFrame, index):
    row = data.iloc[[index]]
    sample = row.values[0][:-1], row.values[0][-1]
    return sample
