import math
import matplotlib.pyplot as plt


def get_grid_dimensions(n_features, ncols=2):
    """
    Calculate the number of rows and columns needed for a subplot grid.

    Parameters
    ---
    n_features : int
        Total number of features to plot.
    ncols : int, optional
        Number of columns in the subplot grid. Defaults to 2.

    Returns
    ---
    tuple
        A tuple (nrows, ncols) representing the number of rows and columns
        for the subplot layout.
    """
    nrows = math.ceil(n_features / ncols)
    return nrows, ncols


def setup_figure(nrows, ncols, row_height=4, base_width=12, title=None, fontsize=16):
    """
    Initialize a figure for multiple subplots.

    Parameters
    ---
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    row_height : int, optional
        Height of each row in the figure. Defaults to 4.
    base_width : int, optional
        Total width of the figure. Defaults to 12.
    title : str, optional
        Suptitle for the entire figure.
    fontsize : int, optional
        Font size for the title. Defaults to 16.
    """
    plt.figure(figsize=(base_width, row_height * nrows))
    if title:
        plt.suptitle(title, fontsize=fontsize)


def setup_single_plot(figsize=(7, 5), title=None):
    """
    Initialize a figure for a single plot.

    Parameters
    ---
    figsize : tuple, optional
        Size of the figure. Defaults to (7, 5).
    title : str, optional
        Title of the plot.
    """
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
