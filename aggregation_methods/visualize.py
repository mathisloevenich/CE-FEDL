import pandas as pd
import bokeh
from bokeh.plotting import figure, show, gridplot
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, central=True, distributed=True):
    acc_figure = figure(title="Accuracy", x_axis_label='rounds', y_axis_label='accuracy')
    loss_figure = figure(title="Loss", x_axis_label='rounds', y_axis_label='loss')

    for label, color, file_path in results:
        df = pd.read_pickle(file_path)

        if central:
            cen_acc = np.array(df["history"].metrics_centralized["accuracy"])
            cen_loss = np.array(df["history"].losses_centralized)
            acc_figure.line(cen_acc[:, 0], cen_acc[:, 1], legend_label="Test Accuracy " + label, line_width=2, color="red")
            loss_figure.line(cen_loss[:, 0], cen_loss[:, 1], legend_label="Test Loss " + label, line_width=2, color="red")

        if distributed:
            dis_acc = np.array(df["history"].metrics_distributed["accuracy"])
            dis_loss = np.array(df["history"].losses_distributed)
            acc_figure.line(dis_acc[:, 0], dis_acc[:, 1], legend_label="Val Accuracy " + label, line_width=2, color="black")
            loss_figure.line(dis_loss[:, 0], dis_loss[:, 1], legend_label="Val Loss " + label, line_width=2, color="black")

    acc_figure.legend.location = "bottom_right"
    grid = gridplot([[acc_figure, loss_figure]])
    show(grid)


if __name__ == "__main__":
    file_paths = [("FedAvg", "red", r".\results\femnist_resnet18_c200.pkl")]
    plot_results(file_paths, central=False)
