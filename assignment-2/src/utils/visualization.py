from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import src.utils.feat_engineering as fe


def missing_values_plot(
        data: pd.DataFrame,
        subplot_size: (int, int),
        title: str = "", width: int = 7,
        mode: str = "pct",
        **barplot_kwargs
) -> (plt.Figure, np.ndarray):
    """
    Plots a grid of barplots, each representing missing value information about each feature in the
    provided dataframe.

    :param data: data to plot missing values information for
    :param subplot_size: dimension of each subplot in the grid
    :param title: title of the (whole) plot
    :param width: width (in plots) of the grid
    :param mode: either "pct" or "count". If pct, plots missing%, else missing count
    :param barplot_kwargs: additional parameter to pass to the underlying barplot function
    :return: (fig, axs) a grid of barplots about the numbers/percentages of missing values
        for each feature in the provided dataframe
    """

    # Arg check
    PCT_MODE = "pct"
    COUNT_MODE = "count"
    if mode != PCT_MODE and mode != COUNT_MODE:
        raise Exception(f"mode parameter should be '{PCT_MODE}' or '{COUNT_MODE}', got {mode}")

    # Get dataframe containing missing values information
    missing_df = fe.get_missing_info(df=data)
    features_col = missing_df["column"]
    pct_col = missing_df["missing %"]
    count_col = missing_df["missing count"]

    # Create plotting grid
    n_features = len(data.columns)
    fig, axs = _get_plotting_grid(width, n_features, subplot_size, style="whitegrid")

    # Create a plot for each grid square
    plot_row = 0
    for i, feature in enumerate(data.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        plot_onto = axs[plot_row, plot_col]

        # Mask used to extract values belonging to the current feature
        # from the missing info dataframe
        current_feature_mask = features_col == feature

        # Get the current feature missing information
        if mode == PCT_MODE:
            missing_info = pct_col[current_feature_mask]
            plot_onto.set_ylim([0, 100])

            # Show tick every 10%
            plot_onto.set_yticks(list(range(0, 101, 10)))
        else:
            missing_info = count_col[current_feature_mask]
            plot_onto.set_ylim([0, len(data)])

        sns.barplot(ax=plot_onto, x=[feature], y=missing_info, **barplot_kwargs)

    if title != "":
        plt.suptitle(title, fontsize=16)

    return fig, axs


def _get_plotting_grid(
        width: int, tot_cells: int,
        subplot_size: (int, int),
        style: str = "ticks",
        **subplots_kwargs
) -> (plt.Figure, np.ndarray):
    """
    Returns a plot grid based on the provided parameters.

    :param width: width (in plots) of the grid
    :param tot_cells: total number of cells (plots) of the grid
    :param subplot_size: dimension of each subplot in the grid
    :param style: seaborn style of the plots
    :param subplots_kwargs: additional kwargs passed to the underlying pyplot.subplots call
    :return: fig, axs
    """
    sns.set_style(style=style)

    # Calculate dimensions of the grid
    height = tot_cells // width
    if width * height < tot_cells or height == 0:
        height += 1

    fig_width = width * subplot_size[0]
    fig_height = height * subplot_size[1]

    fig, axs = plt.subplots(ncols=width, nrows=height, figsize=(fig_width, fig_height), **subplots_kwargs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)  # Else fig.suptitle overlaps with the subplots

    return fig, axs
