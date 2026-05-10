from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss_curves(
    logs: pd.DataFrame,
    models: List[str],
    figsize_per_model: Tuple[int, int] = (12, 6),
    epoch_limits: Tuple[int, int] = (1, 20),
    loss_limits: Tuple[float, float] = (0, 1),
    palette_name: str = "Set1"
):
    """
    Plot training and validation loss curves for multiple models and return the figure.

    Args:
        logs (pd.DataFrame): Training logs containing columns:
            ['epoch', 'train_loss', 'val_loss', 'model'].
        models (List[str]): List of model names to plot.
        figsize_per_model (Tuple[int, int], optional): Figure size per subplot.
        epoch_limits (Tuple[int, int], optional): (min_epoch, max_epoch) for x-axis.
        loss_limits (Tuple[float, float], optional): (min_loss, max_loss) for y-axis.
        palette_name (str, optional): Seaborn palette name (default 'Set1').

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """

    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize = (figsize_per_model[0], figsize_per_model[1] * num_models))

    if num_models == 1:
        axes = [axes]  # wrap single axis in a list

    palette = sns.color_palette(palette_name, 2)

    for ax, model in zip(axes, models):
        # Filter logs for the selected model
        subframe = logs[logs["model"].str.contains(model.lower(), case=False)]

        # Plot training and validation loss
        sns.lineplot(subframe, x = "epoch", y = "train_loss", hue = "model",
                     palette = palette, ax = ax)
        sns.lineplot(subframe, x = "epoch", y = "val_loss", hue = "model",
                     palette = palette, linestyle = "--", ax = ax)

        # Axis formatting
        ax.set_xlim(epoch_limits)
        ax.set_xticks(list(range(epoch_limits[0] + 1, epoch_limits[1] + 2, 2)))
        ax.set_xticklabels(list(range(epoch_limits[0] + 1, epoch_limits[1] + 2, 2)))

        ax.set_ylim(loss_limits)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Categorical Cross Entropy Loss")
        ax.set_title(f"Loss Curves for {model}")

        # Extract handles for legend
        handles, _ = ax.get_legend_handles_labels()

        # Assign custom legend labels
        labels = [
            "PyTorch Train Loss", "TensorFlow Train Loss",
            "PyTorch Validation Loss", "TensorFlow Validation Loss"
        ]
        ax.legend(handles = handles[:4], labels = labels, loc = "upper right", ncols = 2)

    plt.tight_layout()

    return fig

def plot_confusion_matrix(
    model_name: str,
    framework: str,
    inference_df: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a normalized confusion matrix heatmap for a specific model and framework.

    The function expects `inference_df` to contain rows where each row stores arrays
    of true labels and predicted labels. The row corresponding to the combined
    model identifier (e.g., "resnet50_pytorch") is extracted, and a confusion
    matrix is computed and normalized row-wise.

    Args:
        model_name (str): The architecture name, such as "ResNet50".
        framework (str): The framework identifier, typically "tensorflow" or "pytorch".
        inference_df (pd.DataFrame): Must include columns:
            - 'model': combined model+framework key
            - 'labels': array-like of true labels
            - 'predictions': array-like of predicted labels
        class_names (List[str], optional): Class label names. If not provided,
            class names are automatically generated and ordered based on the
            sorted unique values present in the labels.
        figsize (Tuple[int, int]): Size of the figure if a new figure is created.
        ax (matplotlib.axes.Axes, optional): Axis on which to draw the heatmap.
            If None, a new figure and axis are created.

    Returns:
        matplotlib.axes.Axes:
            The axis containing the confusion matrix heatmap.

    Raises:
        ValueError: If the model entry is not found in `inference_df`.
    """

    # Identify row containing results for this model+framework
    full_name = f"{model_name.lower()}_{framework.lower()}"
    row = inference_df[inference_df["model"] == full_name]
    if row.empty:
        raise ValueError(f"Model '{full_name}' not found in inference dataframe.")

    # Extract true and predicted label arrays
    y_true = eval(row["labels"].iloc[0])
    y_pred = eval(row["predictions"].iloc[0])

    # Compute confusion matrix counts
    cm = confusion_matrix(y_true, y_pred)

    # Build class names automatically if not provided
    if class_names is None:
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [f"Class {c}" for c in unique_classes]

    # Normalize confusion matrix row-wise to create percentages
    cm_normalized = cm.astype(float) / cm.sum(axis = 1, keepdims = True)

    # Create figure/axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    # Draw heatmap
    sns.heatmap(
        cm_normalized,
        annot = True,
        fmt = ".2f",
        cmap = "Blues",
        cbar = False,
        xticklabels = class_names,
        yticklabels = class_names,
        ax = ax
    )

    # Axis labels and title
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"Normalized Confusion Matrix: {model_name} ({framework})")

    return ax

def compare_confusion_matrices(
    model_name: str,
    inference_df: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    Plot normalized confusion matrices for TensorFlow and PyTorch
    versions of the same architecture side by side.

    Args:
        model_name (str): Model architecture name, such as "ResNet50".
        inference_df (pd.DataFrame): DataFrame with columns for model ID,
            labels, and predictions.
        class_names (List[str], optional): Optional class names. If not provided,
            names will be inferred independently within each confusion matrix.
        figsize (Tuple[int, int]): Size of the full figure.

    Returns:
        matplotlib.figure.Figure:
            The full figure.
    """

    # Create subplot layout for TensorFlow and PyTorch comparisons
    fig, axes = plt.subplots(1, 2, figsize = figsize)

    # Process each framework on its own axis
    for ax, fw in zip(axes, ["Tensorflow", "PyTorch"]):
        plot_confusion_matrix(
            model_name = model_name,
            framework = fw,
            inference_df = inference_df,
            class_names = class_names,
            ax = ax
        )

    # Add figure-level title
    fig.suptitle(f"Confusion Matrix Comparison: {model_name}", fontsize = 16)
    fig.tight_layout()

    return fig