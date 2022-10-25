import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_preprocessing as loader

data = pd.read_csv("datasets/trainingset.csv")


def load_data(raw_data):
    data = loader.load(raw_data)
    print("loaded preprocessed data")
    return data


# data = load_data("datasets/trainingset.csv")

def line_plot(x, y):
    data_sorted = data.sort_values(x)
    plt.plot(data_sorted.iloc[:, x], data_sorted.iloc[:, y])

    x_label = f"Feature {x}"
    y_label = f"Feature {y}"
    if y == 19:
        y_label = "Claim Amount"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs " + y_label + " Line Plot")
    plt.show()
    plt.close()


def bar_chart(x):
    categories, counts = np.unique(data.iloc[:, x], return_counts=True)
    plt.bar(categories, counts)

    x_label = f"Feature {x}"
    if x == 19:
        x_label = "Claim Amount"
    plt.xlabel(x_label + "s")
    plt.ylabel("Count")
    plt.title(x_label + " Bar Chart")
    plt.show()
    plt.close()


def scatter_plot(x, y):
    plt.scatter(data.iloc[:, x], data.iloc[:, y])

    x_label = f"Feature {x}"
    y_label = f"Feature {y}"
    if y == 19:
        y_label = "Claim Amount"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs " + y_label + " Scatter Plot")
    plt.show()
    plt.close()


def histogram(x):
    if len(set(data.iloc[:, x])) < 10:  # using set (no duplicates) to see the num of unique values
        plt.hist(data.iloc[:, x], bins=len(set(data.iloc[:, x])))
    else:
        plt.hist(data.iloc[:, x])

    x_label = f"Feature {x}"
    if x == 19:
        x_label = "Claim Amount"
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(x_label + " Histogram")
    plt.show()
    plt.close()


for i in range(data.shape[1]):
    if i != 0:
        histogram(i)
        if i != 19:
            scatter_plot(i, 19)
