# Computer Vision Transfer Learning Using VGG16, ResNet50, and InceptionV3

## Summary

Transfer learning is the process of using a neural network model pre-trained on a similar task to drastically increase the speed and effectiveness of training a new neural network. In the field of computer vision, a pre-trained convolutional neural network (CNN) is trained on a massive and robust dataset to produce a strong generalist model. The final prediction layer of the pre-trained model is stripped away and replaced with additional connecting layers and a new output layer suited to the new task. This technique is known as feature extraction, as the earlier layers of the CNN extract more general features while later layers focus on more specific features that are unique to each particular task. This process makes training a new model swift and dramatically reduces the amount of data needed to train.

This project utilizes the VGG16, ResNet50, and InceptionV3 pre-trained models as backbones for creating new models that can predict the species of a flower from an image. Each model was created using both PyTorch and Tensorflow to demonstrate the differences in training between each library. Image augmentation was used in order to increase model robustness and reduce sensitivity by randomly adjusting the brightness, contrast and rotation of each image and prevent the models from fitting too closely to the provided training data. Each model was trained for 20 epochs and serialized for ease of restoration. The training curves were analyzed to gauge overfitting, and a confusion matrix analysis was performed to identify which classes of flowers were easy or difficult for each model to predict.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/CV-Flowers-Transfer-Learning/blob/main/requirements.txt).

```
keras==3.11.3
matplotlib==3.10.6
numpy==1.26.4
pandas==2.3.3
pillow==11.3.0
scikit-learn==1.7.2
seaborn==0.13.2
tensorflow==2.20.0
torch==2.9.0+cpu
torchinfo==1.8.0
torchvision==0.24.0+cpu
```

## Repository Structure

```
├── data                                           # Raw images grouped by flower and partitioned into train and test datasets
├── logs                                           # Training and validation metrics across training epochs
├── models                                         # Model weights at each epoch of transfer learning under PyTorch and Tensorflow
├── predictions                                    # Model predictions and classification metrics
├── src                                            # Project notebooks and source code
│   ├── Transfer_Learning.ipynb                        # Notebook containing preprocessing, training, and analysis
│   ├── modules                                        # Source code with custom functions
│   │   ├── evaluation.py                                  # Functions for combining training logs and model predictions into readable form
│   │   ├── plotting.py                                    # Functions for Matplotlib and Seaborn plotting
│   │   ├── preprocessing.py                               # Functions for preprocessing images, image augmentation, and dataset creation under PyTorch and Tensorflow
│   │   └── training.py                                    # Functions for training. logging, and saving models under PyTorch and Tensorflow
├── README.md
└── requirements.txt
```
