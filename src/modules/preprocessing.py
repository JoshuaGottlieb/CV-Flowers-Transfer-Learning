import os
import random
import shutil
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import inception_v3, resnet50, vgg16

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---- Train Test Split ----

def train_test_split(
    directories: List[str],
    labels: List[str],
    train_dir: str,
    test_dir: str,
    train_ratio: float = 0.75,
    seed: int = 42
) -> None:
    """
    Split image files from multiple class directories into train and test sets.

    Args:
        directories (List[str]): List of directories, one per class.
        labels (List[str]): List of class labels, same order as directories.
        train_dir (str): Output directory for training images.
        test_dir (str): Output directory for test images.
        train_ratio (float): Fraction of images to put in the train set (0–1).
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)
    random.seed(seed)

    for label, flower_dir in zip(labels, directories):
        paths = os.listdir(flower_dir)
        random.shuffle(paths)

        train_num = int(len(paths) * train_ratio)

        for i, filename in enumerate(paths, start = 1):
            # Decide target directory
            target_dir = train_dir if i <= train_num else test_dir

            # Copy file
            shutil.copy(
                os.path.join(flower_dir, filename),
                os.path.join(target_dir, f"{label}_{i:03d}.jpg")
            )

    return

# ---- Utilities ----

def label_from_filename(filename: str) -> str:
    """
    Extract a class label from an image filename.

    This function assumes filenames follow the pattern
    `label_####.ext` (e.g., `tulip_984.jpg`). The class label is taken
    as the substring before the first underscore.

    Args:
        filename (str): Full or relative path to the image file.

    Returns:
        str: The extracted label (substring before the first underscore).

    Raises:
        ValueError: If the filename does not contain an underscore,
            making label extraction impossible.
    """
    # Get the last component of the path (e.g., "/path/to/tulip_984.jpg" -> "tulip_984.jpg")
    base: str = os.path.basename(filename)

    # Ensure the filename contains an underscore to split on
    if "_" not in base:
        raise ValueError(
            f"Filename '{base}' does not contain an underscore; "
            "cannot extract label."
        )

    # Split at the first underscore and return the left part as the label
    label: str = base.split("_", 1)[0]

    return label

def get_image_size(architecture: str) -> Tuple[int, int]:
    """
    Return the expected input image size for a given neural network architecture.

    This function maps commonly used model architectures to the required
    input image dimensions. These sizes are typically determined by the
    architecture's design or its pretrained weights.

    Args:
        architecture (str): Name of the model architecture
            (e.g., "inceptionv3", "resnet50", "vgg16").

    Returns:
        Tuple[int, int]: The required (height, width) of input images.

    Raises:
        ValueError: If the specified architecture is not supported.
    """
    architecture = architecture.lower()

    # InceptionV3 requires 299×299 inputs
    if architecture == "inceptionv3":
        return (299, 299)

    # ResNet50 and VGG16 both use 224×224 inputs
    elif architecture in ["resnet50", "vgg16"]:
        return (224, 224)

    # Any other architecture is unsupported in this mapping
    else:
        raise ValueError(f"Unsupported architecture '{architecture}'.")


# ---- Augmentations and Preprocessing ----
        
def get_augmentations_tf() -> tf.keras.Sequential:
    """
    Create a TensorFlow/Keras data augmentation pipeline.

    The returned augmentation model applies deterministic, GPU-accelerated
    color-space and geometric transformations. All augmentations operate
    on batches produced by tf.data pipelines.

    Augmentations applied:
        - Random horizontal flip
        - Random rotation (±12.5%)
        - Random brightness adjustment
        - Random contrast adjustment

    Returns:
        tf.keras.Sequential: A Sequential model containing image
        augmentation layers.
    """
    # Sequential container applying preprocessing on the GPU
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),        # Random left-right flipping
        tf.keras.layers.RandomRotation(0.125),           # Rotate by ±45 degrees
        tf.keras.layers.RandomBrightness(factor = 0.2),  # Adjust brightness
        tf.keras.layers.RandomContrast(factor = 0.2),    # Adjust contrast
    ])


def get_augmentations_torch(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Create a PyTorch augmentation pipeline for training image models.

    This pipeline includes geometric transformations and color-space
    variations. It also ensures all images are resized to the desired
    model input size.

    Args:
        img_size (Tuple[int, int]): Target (height, width) used to resize
            images before augmentation.

    Returns:
        transforms.Compose: A composition of torchvision transforms
        applied during dataset loading.
    """
    return transforms.Compose([
        transforms.Resize(img_size),                     # Ensure uniform image size
        transforms.RandomHorizontalFlip(),               # Random left-right flipping
        transforms.RandomRotation(45),                   # Rotate by ±45 degrees
        transforms.ColorJitter(brightness = 0.2,
                               contrast = 0.2),          # Adjust brightness/contrast
    ])

def get_preprocessing(architecture: str, framework: str) -> Union[Callable, transforms.Normalize]:
    """
    Retrieve the appropriate preprocessing function or transform for a given
    model architecture and deep learning framework.

    For TensorFlow/Keras, this returns the architecture-specific
    `preprocess_input` function from `tf.keras.applications`. For PyTorch,
    this returns a `transforms.Normalize` instance configured with standard
    ImageNet mean and standard deviation values.

    Args:
        architecture (str): Name of the model architecture, such as
            "resnet50", "inceptionv3", or "vgg16".
        framework (str): Deep learning framework to use: "tensorflow" or "pytorch".

    Returns:
        Union[Callable, transforms.Normalize]:
            - For TensorFlow: A callable preprocessing function.
            - For PyTorch: A torchvision `Normalize` transform.

    Raises:
        ValueError: If an unknown architecture or unsupported framework is provided.
    """
    architecture = architecture.lower()
    framework = framework.lower()

    # TensorFlow preprocessing functions
    if framework == "tensorflow":
        if architecture == "resnet50":
            return resnet50.preprocess_input
        elif architecture == "inceptionv3":
            return inception_v3.preprocess_input
        elif architecture == "vgg16":
            return vgg16.preprocess_input
        else:
            raise ValueError(f"Unknown architecture '{architecture}' for TensorFlow preprocessing.")

    # PyTorch preprocessing using ImageNet normalization
    elif framework == "pytorch":
        if architecture in ["resnet50", "vgg16", "inceptionv3"]:
            return transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        else:
            raise ValueError(f"Unknown architecture '{architecture}' for PyTorch preprocessing.")

    # If the framework selection is invalid
    else:
        raise ValueError("Framework must be 'tensorflow' or 'pytorch'.")

# ---- Dataset Loading ----
        
def load_tensorflow_datasets(
    data_dir: str,
    architecture: str,
    batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load TensorFlow training and testing datasets with preprocessing and augmentation.

    This function constructs `tf.data.Dataset` pipelines for training and testing
    images stored in a directory structure of the form:

        data_dir/
            train/*.jpg
            test/*.jpg

    Images are:
        - Loaded and resized based on the required model input size.
        - Augmented (training only) using TensorFlow preprocessing layers.
        - Normalized using architecture-specific preprocessing.
        - Assigned integer labels derived from filename prefixes
          (e.g., "tulip_984.jpg" → "tulip").

    Args:
        data_dir (str): Path to the root dataset directory containing `train/` and `test/`.
        architecture (str): Model architecture used to determine image size and preprocessing.
        batch_size (int): Number of samples per batch for the output datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
            - train_ds: Training `tf.data.Dataset` with augmentation and preprocessing applied.
            - test_ds: Test `tf.data.Dataset` with preprocessing applied.
            - class_names: Sorted list of class label strings.

    Raises:
        ValueError: If the architecture is unsupported by preprocessing or size functions.
    """

    # Determine required image dimensions (e.g., (224, 224))
    img_size = get_image_size(architecture)

    # Get model-specific preprocessing function (e.g., ResNet50 preprocess_input)
    preprocess = get_preprocessing(architecture, "tensorflow")

    # Data augmentation layers for training
    augment = get_augmentations_tf()

    def load_image(path: tf.Tensor) -> tf.Tensor:
        """
        Read an image from disk, decode JPEG bytes, and resize to the target size.
        """
        img = tf.io.read_file(path)                     # Load JPEG file
        img = tf.image.decode_jpeg(img, channels = 3)   # Decode into RGB tensor
        img = tf.image.resize(img, img_size)            # Resize to model input dimensions
        return img

    # Collect image paths for training and testing splits
    train_files = tf.io.gfile.glob(os.path.join(data_dir, "train", "*.jpg"))
    test_files  = tf.io.gfile.glob(os.path.join(data_dir, "test", "*.jpg"))

    # Extract string labels from filenames
    train_labels = [label_from_filename(p) for p in train_files]
    test_labels  = [label_from_filename(p) for p in test_files]

    # Build a consistent class-name list and assign integer indices
    class_names = sorted(set(train_labels + test_labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Map each string label to its integer index
    train_labels = [class_to_idx[l] for l in train_labels]
    test_labels  = [class_to_idx[l] for l in test_labels]

    # Construct tf.data datasets with file paths and numeric labels
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    test_ds  = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

    def process_train(path: tf.Tensor, label: tf.Tensor):
        """
        Load, augment, and preprocess a training image.
        """
        img = load_image(path)
        img = augment(img)           # Apply augmentation layers
        return preprocess(img), label

    def process_test(path: tf.Tensor, label: tf.Tensor):
        """
        Load and preprocess a test image (no augmentation).
        """
        img = load_image(path)
        return preprocess(img), label

    # Build full training pipeline: shuffle -> map -> batch -> prefetch
    train_ds = (
        train_ds
        .shuffle(len(train_files))                                 # Randomize order
        .map(process_train, num_parallel_calls = tf.data.AUTOTUNE) # Parallel processing
        .batch(batch_size)                                         # Batch samples
        .prefetch(tf.data.AUTOTUNE)                                # Enable I/O pipelining
    )

    # Build test pipeline: map -> batch -> prefetch
    test_ds = (
        test_ds
        .map(process_test, num_parallel_calls = tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, class_names

class FilenameDataset(Dataset):
    """
    PyTorch Dataset for loading images from file paths with optional
    preprocessing and augmentation.

    Each image is expected to have a filename starting with its class label
    (e.g., "tulip_984.jpg"). Labels are converted to integer indices using
    a provided class-to-index mapping.

    Args:
        filepaths (List[str]): List of image file paths.
        preprocess (Callable): Function or torchvision transform to preprocess images.
        augment (Optional[Callable]): Optional function or transform for data augmentation.
        img_size (Tuple[int, int]): Target image size (height, width) for resizing.
        class_to_idx (Dict[str, int]): Mapping from class name to integer label.
    """

    def __init__(
        self,
        filepaths: List[str],
        preprocess: Callable,
        augment: Optional[Callable],
        img_size: Tuple[int, int],
        class_to_idx: Dict[str, int]
    ):
        self.filepaths = filepaths
        self.preprocess = preprocess
        self.augment = augment
        self.img_size = img_size
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single image and its label.

        Steps:
            1. Retrieve the file path for the given index.
            2. Extract the class label string from the filename.
            3. Convert the label string to an integer using `class_to_idx`.
            4. Load the image from disk and convert to RGB.
            5. Apply augmentation (if any) and preprocessing transforms.
            6. Return the transformed image and label as a tuple.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Preprocessed image tensor and integer label.
        """
        path = self.filepaths[idx]

        # Extract label string from filename and map to integer
        label_str = label_from_filename(path)
        label = self.class_to_idx[label_str]

        # Load image and ensure 3 channels (RGB)
        img = Image.open(path).convert("RGB")

        # Apply optional augmentation
        if self.augment:
            img = self.augment(img)

        # Apply preprocessing (e.g., normalization, resizing)
        img = self.preprocess(img)

        return img, label

def load_pytorch_datasets(
    data_dir: str,
    architecture: str,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load PyTorch training and testing datasets with preprocessing and augmentation.

    Images are expected to be organized in the following directory structure:

        data_dir/
            train/*.jpg
            test/*.jpg

    Filenames must start with the class label (e.g., "tulip_984.jpg"). Labels
    are automatically extracted and converted to integer indices.

    Args:
        data_dir (str): Root directory containing `train/` and `test/` subdirectories.
        architecture (str): Model architecture to determine input image size and preprocessing.
        batch_size (int): Number of samples per batch for the DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]:
            - train_loader: DataLoader for training dataset with augmentation and preprocessing.
            - test_loader: DataLoader for test dataset with preprocessing only.
            - classes: Sorted list of class label strings.
    """

    # Determine target image size based on architecture
    img_size = get_image_size(architecture)

    # Compose preprocessing transforms: resize -> to tensor -> normalize
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        get_preprocessing(architecture, "pytorch")
    ])

    # Compose augmentation transforms for training
    augment = get_augmentations_torch(img_size)

    # Collect and sort image file paths for training and testing
    train_files = sorted(glob(os.path.join(data_dir, "train", "*.jpg")))
    test_files  = sorted(glob(os.path.join(data_dir, "test", "*.jpg")))

    # Extract string labels from filenames
    train_labels = [label_from_filename(f) for f in train_files]
    test_labels  = [label_from_filename(f) for f in test_files]

    # Build a consistent sorted list of class names and map to integer indices
    classes = sorted(set(train_labels + test_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Create PyTorch Dataset instances
    train_dataset = FilenameDataset(
        train_files, preprocess, augment, img_size, class_to_idx
    )
    test_dataset = FilenameDataset(
        test_files, preprocess, None, img_size, class_to_idx
    )

    # Create DataLoaders for batching and shuffling
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4
    )

    return train_loader, test_loader, classes

def load_image_datasets(
    data_dir: str,
    architecture: str,
    framework: str,
    batch_size: int = 32
) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset, List[str]],
           Tuple[DataLoader, DataLoader, List[str]]]:
    """
    Load image datasets for either TensorFlow or PyTorch with preprocessing 
    and optional augmentation.

    This function provides a unified interface to load training and testing 
    datasets for a specified deep learning framework. The datasets are expected 
    to follow the directory structure:

        data_dir/
            train/*.jpg
            test/*.jpg

    Filenames must start with the class label (e.g., "tulip_984.jpg"). Labels
    are automatically extracted and converted to integer indices.

    Args:
        data_dir (str): Path to the root dataset directory.
        architecture (str): Model architecture name (used to determine input size and preprocessing).
        framework (str): Deep learning framework, either "tensorflow" or "pytorch".
        batch_size (int, optional): Batch size for the returned datasets. Defaults to 32.

    Returns:
        Union[Tuple[tf.data.Dataset, tf.data.Dataset, List[str]],
              Tuple[DataLoader, DataLoader, List[str]]]:
            - For TensorFlow: (train_ds, test_ds, class_names)
            - For PyTorch: (train_loader, test_loader, class_names)

    Raises:
        ValueError: If an unsupported framework is specified.
    """
    framework = framework.lower()

    # Load datasets for TensorFlow
    if framework == "tensorflow":
        return load_tensorflow_datasets(
            data_dir, architecture, batch_size
        )

    # Load datasets for PyTorch
    elif framework == "pytorch":
        return load_pytorch_datasets(
            data_dir, architecture, batch_size
        )

    # Raise error if framework is not recognized
    else:
        raise ValueError("framework must be 'tensorflow' or 'pytorch'.")