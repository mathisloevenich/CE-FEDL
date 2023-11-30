import pandas as pd
import bokeh
from bokeh.plotting import figure, show, gridplot
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    acc_figure = figure(title="Centralized Accuracy", x_axis_label='rounds', y_axis_label='accuracy')
    loss_figure = figure(title="Centralized Loss", x_axis_label='rounds', y_axis_label='loss')

    for label, color, file_path in results:
        df = pd.read_pickle(file_path)
        acc = np.array(df["history"].metrics_centralized["accuracy"])
        loss = np.array(df["history"].losses_centralized)

        acc_figure.line(acc[:, 0], acc[:, 1], legend_label="Accuracy " + label, line_width=2, color=color)
        loss_figure.line(loss[:, 0], loss[:, 1], legend_label="Loss " + label, line_width=2, color=color)

    acc_figure.legend.location = "bottom_right"
    grid = gridplot([[acc_figure, loss_figure]])
    show(grid)


if __name__ == "__main__":
    file_paths = [("FedAvg", "red", r".\results\first_test\fed.pkl"),
                  ("FedAvgM", "blue", r".\results\first_test\fed_m04.pkl")]
    plot_results(file_paths)
