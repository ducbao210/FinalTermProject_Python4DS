# Color constants
BOXPLOT_COLOR = "skyblue"
COUNTPLOT_COLOR = "lightgreen"
HISTOGRAM_COLOR = "coral"
KDE_COLOR = "purple"
SCATTER_EDGE_COLOR = "white"
PIECHART_COLORS = [
    "#F0E491",
    "#BBC863",
    "#658C58",
    "#31694E",
    "#3F4E4F",
    "#2C3639",
    "#3C467B",
    "#50589C",
    "#636CCB",
    "#6E8CFB",
    "#1B3C53",
    "#8D5F8C",
    "#6B3F69",
    "#A1C2BD",
]
# Grid settings
GRID_ALPHA = 0.7
GRID_STYLE = "--"

import matplotlib.pyplot as plt


def apply_grid(axis="y", zorder=1):
    """
    Apply a styled grid to the current plot.

    Parameters
    ---
    axis : str, optional
        Orientation of the grid. Options are 'x', 'y', or 'both'. Defaults to 'y'.
    zorder : int, optional
        Z-order level for the grid, determining its stacking order. Defaults to 1.
    """
    plt.grid(axis=axis, linestyle=GRID_STYLE, alpha=GRID_ALPHA, zorder=zorder)
