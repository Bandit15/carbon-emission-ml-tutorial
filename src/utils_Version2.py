"""
Utility Functions for Carbon Emission ML Tutorial

This module contains helper functions that can be reused across scripts and notebooks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(directory):
    """
    Ensure that a directory exists. If not, create it.

    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plot(fig, filename, directory="../figures"):
    """
    Save a matplotlib figure to a directory.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        filename (str): The filename to use.
        directory (str): The directory to save the figure in.
    """
    ensure_dir(directory)
    fig.savefig(os.path.join(directory, filename), bbox_inches='tight')

def print_evaluation(y_true, y_pred, model_name="Model"):
    """
    Print RMSE and R2 for a regression model.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        model_name (str): Name of the model.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Results:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2:.3f}")

def display_feature_importance(importances, feature_names, top_n=10):
    """
    Display and plot the top_n feature importances.

    Args:
        importances (array-like): Feature importances (e.g., from tree-based models).
        feature_names (list): Names of the features.
        top_n (int): Number of top features to display.
    """
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    print(imp_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(imp_df['Feature'], imp_df['Importance'], color='orange')
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_ylabel("Importance")
    ax.set_xticklabels(imp_df['Feature'], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()