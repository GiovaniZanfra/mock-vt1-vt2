import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.general import compute_mae_by_group


def plot_residuals_by_category(y_test, y_pred, categories):
    """
    Plots boxplots of residuals by category to see if the model systematically under/over-predicts for certain groups.

    Parameters:
    - y_test: array-like of true values
    - y_pred: array-like of predicted values
    - categories: array-like of categorical labels
    """
    # Convert inputs to numpy arrays
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()
    categories = np.array(categories)

    # Calculate residuals
    residuals = y_test - y_pred

    # Prepare figure
    fig, ax = plt.subplots()
    # Plot boxplots of residuals by category
    sns.boxplot(x=categories, y=residuals, ax=ax)

    # Labels and title
    ax.set_xlabel("Categories")
    ax.set_ylabel("Residuals (y_test - y_pred)")
    ax.set_title("Boxplots of Residuals by Category")

    plt.tight_layout()
    return fig


def plot_residuals_hist_kde(y_test, y_pred):
    """
    Plots a histogram and KDE of residuals to check if errors are roughly Gaussian or skewed.

    Parameters:
    - y_test: array-like of true values
    - y_pred: array-like of predicted values
    """
    # Convert inputs to numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate residuals
    residuals = y_test - y_pred

    # Prepare figure
    fig, ax = plt.subplots()

    # Plot histogram and KDE
    sns.histplot(residuals, kde=True, ax=ax, stat="density")

    # Labels and title
    ax.set_xlabel("Residuals (y_test - y_pred)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram and KDE of Residuals")

    plt.tight_layout()
    return fig


def plot_residuals_vs_predicted(y_test, y_pred):
    """
    Plots a scatter plot of residuals against predicted values to check for heteroscedasticity or non-linear patterns.

    Parameters:
    - y_test: array-like of true values
    - y_pred: array-like of predicted values
    """
    # Convert inputs to numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate residuals
    residuals = y_test - y_pred

    # Prepare figure
    fig, ax = plt.subplots()

    # Scatter plot of residuals vs predicted values
    ax.scatter(y_pred, residuals, alpha=0.6)

    # Add horizontal line at y=0 (ideal residual value)
    ax.axhline(y=0, color="r", linestyle="--", label="Residual = 0")

    # Labels and legend
    ax.set_xlabel("Predicted values (y_pred)")
    ax.set_ylabel("Residuals (y_test - y_pred)")
    ax.set_title("Residuals vs Predicted Values")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_grouped_mae(X, y_true, y_pred, group_cols, title, save_path=None):
    df = compute_mae_by_group(X, y_true, y_pred, group_cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["subgroup"], df["mae"])
    ax.set_title(title)
    ax.set_ylabel("MAE")
    ax.set_xlabel("Group")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_pred_vs_true(y_test, y_pred, categories=None):
    """
    Plots a scatter of y_test vs y_pred with error bounds at ±10% and ±20%,
    a fitted regression line, and axes starting at zero.
    Uses softer colors and thinner lines for clarity.

    Parameters:
    - y_test: array‐like of true values
    - y_pred: array‐like of predicted values
    - categories: array‐like of categorical labels (optional)
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    max_val = max(y_test.max(), y_pred.max())
    x = np.linspace(0, max_val, 100)

    fig, ax = plt.subplots()

    # scatter
    if categories is not None:
        categories = np.array(categories)
        for cat in np.unique(categories):
            mask = categories == cat
            ax.scatter(y_test[mask], y_pred[mask], label=str(cat), alpha=0.5, s=20)
    else:
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="slategray")

    # identity line (perfect prediction)
    ax.plot(x, x, linestyle="--", linewidth=1.0, color="lightgray", label="Perfect")

    # error bounds ±10% and ±20%
    ax.plot(x, x * 1.10, linestyle=":", linewidth=0.8, color="lightblue", label="+10%")
    ax.plot(x, x * 0.90, linestyle=":", linewidth=0.8, color="lightblue", label="-10%")
    ax.plot(
        x, x * 1.20, linestyle="-.", linewidth=0.8, color="lightcoral", label="+20%"
    )
    ax.plot(
        x, x * 0.80, linestyle="-.", linewidth=0.8, color="lightcoral", label="-20%"
    )

    # regression line
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    reg_line = slope * x + intercept
    ax.plot(
        x,
        reg_line,
        linewidth=1.2,
        color="navy",
        label=f"Fit: y={slope:.2f}x+{intercept:.2f}",
    )

    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_xlim(0, max_val * 1.02)
    ax.set_ylim(0, max_val * 1.02)
    ax.set_title("Predicted vs True with Regression & Error Bounds")
    ax.legend(frameon=False, fontsize="small")

    plt.tight_layout()
    return fig
