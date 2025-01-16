# HairClassification Project

## Overview
The HairClassification project is an project focused on classifying different types of hair, such as curly, straight, and wavy. This project leverages the power of machine learning and deep learning techniques to achieve accurate classification. The repository includes multiple scripts for essential tasks, such as data processing and model definition.

## Contains scripts

### `dataProcessing.py`

Resizes and augments images to prepare them for training a machine learning model. It checks the validity of image files, resizes them to a specified size, applies augmentations (shearing, zooming, horizontal flip) to a subset of images, and saves the results to an output directory.

### `cnn_model.py` and `CNN_model.ipynb`

The script is used to train a convolutional neural network (CNN) model for image classification. It loads and preprocesses the images, splitting them into training and test sets, normalizing them, and applying one-hot encoding to the labels. It uses data augmentation to generate variations of the training images, and loads the pre-trained VGG16 model to extract features. It adds additional layers to VGG16, compiles the model, and trains it. Evaluates the model on the test dataset, displays the accuracy plot and confusion matrix to visualize the classification performance.

### `cnn_methods.py`

The script defines functions for fundamental operations of convolutional neural networks (CNN). It includes functions for performing 2D convolutions (`convolution`), applying max pooling (`max_pool`), flattening tensors (`flatten`), and implementing dense layers (`dense`) with activation functions such as ReLU and softmax. It also calculates the loss using cross-categorical entropy (`categorical_crossentropy`). These functions are the basic building blocks for building and training a CNN.

### `cnn_model2.py`

Defines and trains a convolutional neural network (CNN) model using the `tf.distribute.MirroredStrategy` distributed strategy to run on multiple GPUs. Includes functions for generating training and validation datasets, initializing weights, and building the CNN model architecture. Sets environment variables for parallelism and trains the model using machine learning, tracking training and validation losses. Finally, displays the loss graph to visualize the model's performance. **This script is not yet fully functional.**
