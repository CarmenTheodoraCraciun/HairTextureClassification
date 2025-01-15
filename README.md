# HairClassification Project

## Overview
The HairClassification project is an exciting endeavor focused on classifying different types of hair, such as curly, dreadlocks, kinky, straight, and wavy. This project leverages the power of machine learning and deep learning techniques to achieve accurate classification. The repository includes multiple scripts for essential tasks, such as data processing, model definition, and specialized methods for Convolutional Neural Networks (CNNs). Whether you are a data scientist, machine learning enthusiast, or simply someone interested in image classification, this project provides a comprehensive framework to get started and contribute.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.x installed
- Required libraries: numpy, pandas, sklearn, tensorflow, keras

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CarmenTheodoraCraciun/HairClasification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd HairClasification
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Scripts Description

### 1. `dataProcessing.py`
This script is crucial for preparing the data before feeding it into the machine learning model. Key tasks include:
- Handling Missing Values: Utilizes imputation methods to replace missing data points, ensuring the dataset is complete and accurate.
- Categorical Data Handling: Converts categorical features into numerical values through various encoding techniques, making the data suitable for machine learning algorithms.
- Data Splitting: Divides the dataset into training and testing subsets, which is essential for evaluating the model's performance and generalization capability. This step helps in creating a robust model by preventing overfitting.

### 2. `model3.py`
This script defines the architecture of the deep learning model used for hair classification. It incorporates advanced techniques to accurately classify hair types. Key components include:
- EncoderCNN: Utilizes Convolutional Neural Network (CNN) layers to extract rich feature vectors from input images. These features capture essential details required for classification.
- DecoderRNN: Employs a Recurrent Neural Network (RNN) to generate descriptions or labels for images based on the feature vectors extracted by the EncoderCNN. This combination enhances the model's ability to understand and classify images accurately.

### 3. `cnn_methods.py`
This script contains specialized methods for working with Convolutional Neural Networks (CNNs) in this project. Key functionalities include:
- Feature Extraction: Uses CNNs to extract relevant features from input images, enabling the model to focus on important patterns and details that distinguish different hair types.
- Model Training: Facilitates the training process of the CNN using the available training data. It includes techniques for optimizing the model's performance, such as adjusting hyperparameters and using appropriate loss functions.

## Usage
To run the scripts, follow these steps:
1. Preprocess the data using `dataProcessing.py`:
   ```bash
   python dataProcessing.py
   ```
2. Train the model using `model3.py`:
   ```bash
   python model3.py
   ```
3. Use the methods in `cnn_methods.py` for further feature extraction and model training as needed:
   ```bash
   python cnn_methods.py
   ```
