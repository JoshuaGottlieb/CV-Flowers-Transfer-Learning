import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Custom function for loading images into PyTorch/Tensorflow format
from modules.preprocessing import load_image_datasets

# Custom function for compiling models
from modules.training import compile_model

def combine_logs(log_dir: str) -> pd.DataFrame:
    """
    Combine training logs from multiple model subdirectories into a single DataFrame.

    Each subdirectory inside `log_dir` is expected to contain a `training_log.csv`.
    The function normalizes column names:
      - Ensures `epoch` starts at 1
      - Renames `loss` → `train_loss` when necessary
      - Ensures output columns are ['epoch', 'model', 'train_loss', 'val_loss']

    Args:
        log_dir (str): Path containing model subdirectories with training logs.

    Returns:
        pd.DataFrame: Combined log DataFrame with columns:
            ['epoch', 'model', 'train_loss', 'val_loss']
    """
    logs = pd.DataFrame()

    # Iterate through subdirectories
    for sub in os.listdir(log_dir):
        model_path = os.path.join(log_dir, sub, "training_log.csv")

        if not os.path.isfile(model_path):
            continue  # skip anything that doesn't have a log file

        temp = pd.read_csv(model_path)
        temp["model"] = sub

        # Standardize column names
        if "loss" in temp.columns:
            # TensorFlow style log
            temp["epoch"] = temp["epoch"] + 1
            temp = temp[["epoch", "model", "loss", "val_loss"]]
            temp = temp.rename({"loss": "train_loss"}, axis = 1)
        else:
            # PyTorch style log
            temp = temp[["epoch", "model", "train_loss", "val_loss"]]

        logs = pd.concat([logs, temp], axis = 0, ignore_index = True)

    return logs

def get_predictions_and_metrics(
    best_models: List[Tuple[str, str, int]],
    data_dir: str,
    models_dir: str
) -> pd.DataFrame:
    """
    Run inference for each (architecture, framework, epoch) combination,
    collect predictions, compute metrics, and return a summary DataFrame.

    Args:
        best_models (List[Tuple[str, str, int]]):
            List of triplets (architecture, framework, epoch).
        data_dir (str): Directory containing training/testing images.
        models_dir (str): Directory containing saved models.

    Returns:
        pd.DataFrame: Summary of predictions, labels, and metrics per model.
    """

    inference_data = []
    y_true = []  # Collected once from the first model's test loader

    for architecture, framework, epoch in best_models:

        # Load datasets (we only use test_loader here)
        _, test_loader, _ = load_image_datasets(
            data_dir = data_dir,
            architecture = architecture,
            framework = framework
        )

        # Extract ground-truth labels once
        if not y_true:
            y_true = list(
                np.concatenate([labels for _, labels in test_loader], axis = 0)
            )

        # TensorFlow model inference
        if framework == "tensorflow":
            model_path = os.path.join(
                models_dir,
                architecture,
                f"{architecture}_{framework}-{epoch:03d}.keras"
            )

            # Load Keras model
            model = tf.keras.models.load_model(model_path)

            # Predict probabilities for each test batch
            probabilities = model.predict(test_loader, verbose = 0)

            # Convert probabilities to predicted class IDs
            y_pred = list(np.argmax(probabilities, axis = 1))

        # PyTorch model inference
        elif framework == "pytorch":
            model_path = os.path.join(
                models_dir,
                architecture,
                f"{architecture}_{framework}-{epoch:03d}.pt"
            )

            # Build and load model weights
            model = compile_model(architecture = architecture, framework = framework)
            model.load_state_dict(
                torch.load(model_path, map_location = "cpu", weights_only = True)
            )
            model.eval()

            # Store logits from each batch
            logits_list = []

            with torch.no_grad():
                for inputs, _ in test_loader:
                    logits = model(inputs)
                    logits_list.append(logits)

            # Combine logits into a single tensor
            final_logits = torch.cat(logits_list, dim = 0)

            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(final_logits, dim = 1)

            # Convert to predicted class IDs
            y_pred = list(torch.argmax(probs, dim = 1).numpy())

        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # Compute evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average = "macro")
        recall = recall_score(y_true, y_pred, average = "macro")
        f1 = f1_score(y_true, y_pred, average = "macro")

        # Save metrics and predictions for this model
        inference_data.append({
            "model": f"{architecture}_{framework}",
            "labels": y_true,
            "predictions": y_pred,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # Return full metrics summary
    return pd.DataFrame(inference_data)