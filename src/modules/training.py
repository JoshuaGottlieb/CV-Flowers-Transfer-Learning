import os
from typing import Any, Dict, List, Union

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.callbacks import CSVLogger, History, ModelCheckpoint
from tensorflow.keras.models import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import (
    Inception_V3_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    inception_v3,
    resnet50,
    vgg16,
)
from tqdm import tqdm

# Custom function for determining the image size for each model
from modules.preprocessing import get_image_size


# ---- Build and Compile Models ----

def compile_tf_model(architecture: str) -> tf.keras.Model:
    """
    Build and compile a TensorFlow Keras model with a pretrained backbone 
    and custom classification head.

    The function supports 'resnet50', 'vgg16', and 'inceptionv3' as backbone 
    architectures. The pretrained layers are frozen, and a custom classification 
    head is added for 5-class classification.

    Args:
        architecture (str): Name of the backbone architecture 
            ('resnet50', 'vgg16', 'inceptionv3').

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.

    Raises:
        ValueError: If an unsupported architecture is provided.
    """

    # Select the appropriate pretrained model
    arch = architecture.lower()
    if arch == 'resnet50':
        pretrained = ResNet50
    elif arch == 'vgg16':
        pretrained = VGG16
    elif arch == 'inceptionv3':
        pretrained = InceptionV3
    else:
        raise ValueError(f"Unsupported architecture '{architecture}'")

    # Determine required input size for the chosen architecture
    image_size = get_image_size(architecture)

    # Load the pretrained base model without the top classification layer
    base_model = pretrained(
        weights = 'imagenet',
        include_top = False,
        input_shape = (image_size[0], image_size[1], 3)
    )

    # Freeze all pretrained layers to retain learned features
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dense(5, activation = 'softmax')(x)

    # Build and compile the final model
    model = Model(inputs = base_model.input, outputs = x)
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

def compile_pytorch_model(architecture: str) -> nn.Module:
    """
    Build a PyTorch model with a pretrained backbone and a custom classification head.

    Supports 'resnet50', 'vgg16', and 'inceptionv3' as backbone architectures. 
    Pretrained layers are frozen, and a new fully connected head is added 
    for 5-class classification.

    Args:
        architecture (str): Name of the backbone architecture 
            ('resnet50', 'vgg16', 'inceptionv3').

    Returns:
        nn.Module: PyTorch model ready for training with frozen backbone 
                   and custom classification head.

    Raises:
        ValueError: If an unsupported architecture is provided.
    """

    # Normalize architecture name and select pretrained constructor + weight enum
    arch = architecture.lower()
    if arch == 'resnet50':
        pretrained = resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
    elif arch == 'vgg16':
        pretrained = vgg16
        weights = VGG16_Weights.IMAGENET1K_V1
    elif arch == 'inceptionv3':
        pretrained = inception_v3
        weights = Inception_V3_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported architecture '{architecture}'")

    # Load the chosen pretrained backbone
    model = pretrained(weights = weights)

    # For VGG16, the classification layers are in `model.classifier`.
    # For ResNet/Inception, the classification layer is `model.fc`.
    if arch == 'vgg16':
        # Extract input feature size from VGG16's first linear layer
        num_features = model.classifier[0].in_features

        # Remove the pretrained classifier
        model.classifier = nn.Identity()

        # Freeze all pretrained backbone parameters
        for param in model.parameters():
            param.requires_grad = False

        # Attach a new fully connected classifier head
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    else:
        # Extract input feature size from the existing `fc` layer
        num_features = model.fc.in_features

        # Remove the pretrained classification layer
        model.fc = nn.Identity()

        # Freeze backbone weights
        for param in model.parameters():
            param.requires_grad = False

        # Add a new classification head suitable for 5 classes
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    return model

def compile_model(architecture: str, framework: str) -> Union[tf.keras.Model, nn.Module]:
    """
    Compile a deep learning model with a pretrained backbone and custom classification head
    for either TensorFlow or PyTorch based on the specified framework.

    This function automatically selects the appropriate compile function:
        - TensorFlow: `compile_tf_model`
        - PyTorch: `compile_pytorch_model`

    Args:
        architecture (str): Name of the backbone architecture 
            ('resnet50', 'vgg16', 'inceptionv3').
        framework (str): Deep learning framework, either 'tensorflow' or 'pytorch'.

    Returns:
        Union[tf.keras.Model, nn.Module]: A compiled model ready for training.

    Raises:
        ValueError: If an unsupported framework is specified.
    """
    fw = framework.lower()

    if fw == 'tensorflow':
        return compile_tf_model(architecture)
    elif fw == 'pytorch':
        return compile_pytorch_model(architecture)
    else:
        raise ValueError("Framework must be 'tensorflow' or 'pytorch'.")

# ---- Torch Logging Class ----

class TorchCSVLogger:
    """
    Logs training metrics per epoch to a CSV file, similar to Keras' CSVLogger.

    Records metrics such as training and validation loss for each epoch
    and appends them to a CSV file. Creates the directory if it does not exist
    and writes headers only once.

    Args:
        filename (str):
            Path to the CSV file where metrics will be saved. If the directory
            does not exist, it will be created automatically.

    Attributes:
        filename (str): Path to the CSV file.
        columns_written (bool): Tracks whether CSV header has already been written.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        # Tracks if header has been written to CSV
        self.columns_written = False

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Log metrics for a single epoch to the CSV file.

        Appends a new row with epoch number, training loss, and validation loss.
        Writes CSV headers only once on the first write.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
        """
        # Create a one-row DataFrame for current epoch metrics
        df = pd.DataFrame([[epoch, train_loss, val_loss]],
                          columns = ['epoch', 'train_loss', 'val_loss'])
        # Append to CSV file, write header only if not already written
        df.to_csv(self.filename, mode = 'a', header = not self.columns_written, index = False)
        self.columns_written = True

        return

# ---- Training Functions ----

def fit_tf_model(
    model: Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int,
    model_type: str,
    model_name: str
) -> History:
    """
    Train a TensorFlow Keras model using given training and validation datasets.

    The function sets up CSV logging and model checkpointing, then trains
    the model on the provided `tf.data.Dataset` objects.

    Args:
        model (Model): Compiled TensorFlow Keras model to train.
        train_dataset (tf.data.Dataset): Training dataset, already batched and preprocessed.
        val_dataset (tf.data.Dataset): Validation dataset, already batched and preprocessed.
        epochs (int): Number of epochs to train.
        model_type (str): High-level model type, used in checkpoint path.
        model_name (str): Name of the model, used in checkpoint and CSV log filenames.

    Returns:
        History: Keras History object containing training and validation metrics per epoch.
    """

    # Ensure logging and checkpoint directories exist
    os.makedirs(f'../logs/{model_name}', exist_ok = True)
    os.makedirs(f'../models/{model_type}', exist_ok = True)

    # Setup callbacks for CSV logging and model checkpointing
    callbacks = [
        CSVLogger(filename = f'../logs/{model_name}/training_log.csv'),
        ModelCheckpoint(
            filepath = f'../models/{model_type}/{model_name}-{{epoch:03d}}.keras',
            monitor = 'val_loss',
            save_best_only = False,
            save_weights_only = False,
            save_freq = 'epoch'
        )
    ]
    
    # Train the model using TensorFlow Dataset objects
    history = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs = epochs,
        callbacks = callbacks,
        verbose = 1
    )

    # Return the training history object
    return history

def fit_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    model_type: str,
    model_name: str,
    initial_lr: float = 1e-3
) -> Dict[str, List[float]]:
    """
    Train a PyTorch model using provided DataLoader objects for training and validation.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        epochs (int): Number of training epochs.
        model_type (str): High-level model type, used in saving paths.
        model_name (str): Name of the model, used for logging and saving weights.
        initial_lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Returns:
        Dict[str, List[float]]: Dictionary containing lists of training and validation losses per epoch.
    """
    # Ensure logging and checkpoint directories exist
    os.makedirs(f'../logs/{model_name}', exist_ok = True)
    os.makedirs(f'../models/{model_type}', exist_ok = True)


    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr = initial_lr)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize logging utility (assumes a custom CSV logger exists)
    csv_logger = TorchCSVLogger(f'../logs/{model_name}/training_log.csv')

    # Dictionary to store loss history
    history = {'train_loss': [], 'val_loss': []}

    # Main training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Iterate over training batches
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if not isinstance(outputs, torch.Tensor):
                outputs, _ = outputs
                
            loss_value = criterion(outputs, y_batch)
            loss_value.backward()
            optimizer.step()

            # Accumulate batch loss scaled by batch size
            train_loss += loss_value.item() * X_batch.size(0)

        # Compute average training loss
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                if not isinstance(outputs, torch.Tensor):
                    outputs, _ = outputs
                
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # Log epoch metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        csv_logger.log_epoch(epoch + 1, train_loss, val_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save model weights for this epoch
        epoch_path = f'../models/{model_type}/{model_name}-{epoch + 1:03d}.pt'
        os.makedirs(os.path.dirname(epoch_path), exist_ok = True)
        torch.save(model.state_dict(), epoch_path)

    return history

def fit_model(
    model: Union[nn.Module, tf.keras.Model],
    train,
    validation,
    epochs: int,
    model_type: str,
    model_name: str,
    framework: str
) -> Union[Dict[str, List[float]], tf.keras.callbacks.History]:
    """
    Train a model using the appropriate framework-specific training function
    (TensorFlow or PyTorch).

    Args:
        model (nn.Module or tf.keras.Model): The model to train.
        train: Training dataset (PyTorch DataLoader or TensorFlow Dataset).
        validation: Validation dataset (PyTorch DataLoader or TensorFlow Dataset).
        epochs (int): Number of training epochs.
        model_type (str): High-level model type for logging/checkpoint paths.
        model_name (str): Name of the model for logging/checkpoint paths.
        framework (str): Deep learning framework, either 'tensorflow' or 'pytorch'.

    Returns:
        Union[Dict[str, List[float]], tf.keras.callbacks.History]:
            - PyTorch: Dictionary containing training and validation losses.
            - TensorFlow: Keras History object with training metrics.

    Raises:
        ValueError: If an unsupported framework is specified.
    """
    fw = framework.lower()

    if fw == 'tensorflow':
        # Train TensorFlow model
        return fit_tf_model(
            model = model,
            train_dataset = train,
            val_dataset = validation,
            epochs = epochs,
            model_type = model_type,
            model_name = model_name
        )
    elif fw == 'pytorch':
        # Train PyTorch model
        return fit_torch_model(
            model = model,
            train_loader = train,
            val_loader = validation,
            epochs = epochs,
            model_type = model_type,
            model_name = model_name
        )
    else:
        raise ValueError("Framework must be 'tensorflow' or 'pytorch'.")