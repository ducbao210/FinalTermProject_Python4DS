import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
import numpy as np

from .utils_validation import (
    ensure_columns_exist,
    ensure_numeric,
    ensure_categorical_or_low_cardinality,
    to_list,
)
from .utils_layout import get_grid_dimensions, setup_figure, setup_single_plot
from .utils_style import (
    BOXPLOT_COLOR,
    COUNTPLOT_COLOR,
    HISTOGRAM_COLOR,
    KDE_COLOR,
    PIECHART_COLORS,
    apply_grid,
)


def boxplot(df, features, title="Box Plot", orient="h"):
    """
    Plot boxplots for one or multiple numerical features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    features : str or list
        Numerical column(s) to visualize.
    title : str, optional
        Title of the overall figure. Defaults to 'Box Plot'.
    orient : str, optional
        Orientation of the plot ('h' for horizontal, 'v' for vertical). Defaults to 'h'.
    """

    features = to_list(features)
    ensure_columns_exist(df, features)
    ensure_numeric(df, features)

    nrows, ncols = get_grid_dimensions(len(features))
    setup_figure(nrows, ncols, title=title)

    for i, col in enumerate(features, start=1):
        plt.subplot(nrows, ncols, i)
        sns.boxplot(
            data=df,
            x=col if orient == "h" else None,
            y=col if orient == "v" else None,
            orient=orient,
            color=BOXPLOT_COLOR,
            fliersize=5,
            linewidth=1.5,
        )
        plt.title(f"Box Plot of {col}")

    plt.tight_layout()
    plt.show()


def countplot(df, categorical_features, title="Count Plot"):
    """
    Plot count plots for categorical or low-cardinality numerical features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    categorical_features : str or list
        Categorical column(s) to visualize.
    title : str, optional
        Title of the figure. Defaults to 'Count Plot'.
    """
    features = to_list(categorical_features)
    ensure_columns_exist(df, features)
    ensure_categorical_or_low_cardinality(df, features)

    nrows, ncols = get_grid_dimensions(len(features))
    setup_figure(nrows, ncols, title=title)

    for i, col in enumerate(features, start=1):
        plt.subplot(nrows, ncols, i)
        sns.countplot(
            data=df, x=col, zorder=2, color=COUNTPLOT_COLOR, edgecolor="black"
        )
        plt.title(f"Count Plot of {col}")
        apply_grid(axis="y")
        plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()


def piechart(df, features, title="Pie Chart"):
    """
    Generate pie charts for categorical or low-cardinality numerical features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    features : str or list
        Columns to plot as pie charts.
    title : str, optional
        Main title of the figure. Defaults to 'Pie Chart'.
    """
    features = to_list(features)
    ensure_columns_exist(df, features)
    ensure_categorical_or_low_cardinality(df, features)

    nrows, ncols = get_grid_dimensions(len(features))
    # Pie chart needs slightly different figure size, so set directly
    plt.figure(figsize=(8 * ncols, 6 * nrows))
    if title:
        plt.suptitle(title, fontsize=16)

    for i, col in enumerate(features, start=1):
        counts = df[col].value_counts()
        colors = PIECHART_COLORS[: len(counts)]
        if len(counts) > len(PIECHART_COLORS):
            colors = None  # matplotlib automatically chooses colors if missing

        plt.subplot(nrows, ncols, i)
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.legend(
            wedges,
            counts.index,
            title=col,
            loc="best",
            bbox_to_anchor=(1, 0.5, 0.3, 0.1),
        )
        plt.title(f"Pie Chart of {col}")

    plt.tight_layout()
    plt.show()


def scatterplot(df, x, y, title="Scatter Plot"):
    """
    Create a scatter plot between two numerical variables.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    title : str, optional
        Title of the plot. Defaults to 'Scatter Plot'.
    """
    ensure_columns_exist(df, [x, y])
    setup_single_plot(title=title)
    sns.scatterplot(data=df, x=x, y=y, alpha=0.7, edgecolor="white", s=100)
    plt.show()


def residualplot(df, y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals between true and predicted values.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame containing true and predicted columns.
    y_true : str
        Column name of actual values.
    y_pred : str
        Column name of predicted values.
    title : str, optional
        Title of the plot. Defaults to 'Residual Plot'.
    """
    ensure_columns_exist(df, [y_true, y_pred])
    residuals = df[y_true] - df[y_pred]

    setup_single_plot(title=title)
    sns.scatterplot(x=df[y_pred], y=residuals, alpha=0.7)
    plt.axhline(0, color="red", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()


def histogram(df, features, bins=30, title="Histogram", density=False):
    """
    Plot histogram(s) for numerical features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    features : str or list
        Column name(s) to plot.
    bins : int, optional
        Number of bins. Defaults to 30.
    title : str, optional
        Title of the figure. Defaults to 'Histogram'.
    density : bool, optional
        If True, normalize the histogram. Defaults to False.
    """
    features = to_list(features)
    ensure_columns_exist(df, features)
    ensure_numeric(df, features)

    nrows, ncols = get_grid_dimensions(len(features))
    setup_figure(nrows, ncols, title=title)

    for i, col in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        plt.hist(
            df[col].dropna(),
            bins=bins,
            density=density,
            color=HISTOGRAM_COLOR,
            edgecolor="black",
            alpha=0.7,
            zorder=2,
        )
        apply_grid(axis="both")
        plt.title(f"Histogram of {col}")

    plt.tight_layout()
    plt.show()


def heatmap(df, method="pearson", title="Correlation Heatmap", cmap="mako_r"):
    """
    Plot a correlation heatmap of numeric features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall'). Defaults to 'pearson'.
    title : str, optional
        Title of the figure. Defaults to 'Correlation Heatmap'.
    cmap : str, optional
        Colormap for the heatmap. Defaults to 'mako_r'.
    """
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("Heatmap needs at least 2 numeric columns.")

    corr = numeric_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, mask=mask)
    plt.title(f"{title} ({method.capitalize()} Correlation)")
    plt.show()


def linechart(df, x, y, title="Line Chart"):
    """
    Plot a line chart between two variables to show trends.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis.
    title : str, optional
        Title of the figure. Defaults to 'Line Chart'.
    """
    ensure_columns_exist(df, [x, y])
    df_sorted = df.sort_values(by=x)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sorted, x=x, y=y, zorder=2)
    plt.title(title)
    apply_grid(axis="both")
    plt.show()


def KDEplot(df, features, title="KDE Plot"):
    """
    Plot Kernel Density Estimation (KDE) curves for numeric features.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    features : str or list
        Numeric columns to plot.
    title : str, optional
        Title of the figure. Defaults to 'KDE Plot'.
    """
    features = to_list(features)
    ensure_columns_exist(df, features)
    ensure_numeric(df, features)

    nrows, ncols = get_grid_dimensions(len(features))
    setup_figure(nrows, ncols, title=title)

    for i, col in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        sns.kdeplot(df[col], fill=True, alpha=0.4, color=KDE_COLOR)
        plt.title(f"KDE of {col}")

    plt.tight_layout()
    plt.show()


def feature_importance_plot(model, feature_names, title="Feature Importance"):
    """
    Plot feature importance for linear or tree-based models.

    Parameters
    ---
    model : trained ML model
        Model must have coef_ or feature_importances_ attribute.
    feature_names : list
        List of feature names.
    title : str, optional
        Title of the plot. Defaults to 'Feature Importance'.
    """
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        if importance.ndim > 1:
            importance = importance.ravel()
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    elif hasattr(model, "feature_importances_"):
        df_imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )
    else:
        raise ValueError("Model does not have coef_ or feature_importances_.")

    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(df_imp["Feature"], df_imp["Importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def shap_summary_plot(model, X, title="SHAP Summary Plot"):
    """
    Generate a SHAP summary plot for model interpretability.

    Parameters
    ---
    model : trained ML model
        Model to explain.
    X : pd.DataFrame
        Feature matrix.
    title : str, optional
        Title of the figure. Defaults to 'SHAP Summary Plot'.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=True)


def PDP_plot(model, X, features, title="Partial Dependence Plot"):
    """
    Generate Partial Dependence Plots (PDP) to show feature effects.

    Parameters
    ---
    model : trained estimator
        Must support predict().
    X : pd.DataFrame
        Feature matrix.
    features : list
        Feature names or indices to plot.
    title : str, optional
        Title of the figure. Defaults to 'Partial Dependence Plot'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
