# HairClassification Project

## Overview
The HairClassification project is an project focused on classifying different types of hair, such as curly, straight, and wavy. This project leverages the power of machine learning and deep learning techniques to achieve accurate classification. The repository includes multiple scripts for essential tasks, such as data processing and model definition.

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

This script defines the architecture of the deep learning model used for hair classification. It incorporates advanced techniques to accurately classify hair types. Key components include:

#### EncoderCNN
The Encoder CNN extracts relevant features from images.
```python
def EncoderCNN(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(input_img, x)
```
In this function:
- **Input Layer**: Defines the input image size.
- **Convolutional Layers**: Applies multiple convolutional filters to extract features.
- **MaxPooling Layers**: Reduces the dimensionality of the extracted features, preserving essential information.
- **Dense Layer**: Produces a dense feature vector.

#### DecoderRNN
The Decoder RNN generates descriptions for images based on the feature vectors extracted by the encoder.
```python
def DecoderRNN(input_shape, output_shape):
    input_seq = Input(shape=input_shape)
    x = RepeatVector(output_shape[0])(input_seq)
    x = LSTM(256, return_sequences=True)(x)
    x = TimeDistributed(Dense(output_shape[1], activation='softmax'))(x)
    return Model(input_seq, x)
```

#### Full Model
The combination of Encoder-Decoder forms the complete model.
```python
def create_model(input_shape, output_shape):
    encoder = EncoderCNN(input_shape)
    decoder = DecoderRNN(encoder.output_shape[1:], output_shape)
    model_input = encoder.input
    model_output = decoder(encoder.output)
    model = Model(model_input, model_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
In this function:
- **Creating Encoder and Decoder**: Constructs both parts of the model.
- **Connecting the Model**: The input and output of the encoder are connected to the decoder.
- **Compiling the Model**: The model is compiled with an optimizer and appropriate loss function.

### Training the Model
The model can be trained using the `fit` function from Keras.
```python
model = create_model(input_shape=(128, 128, 3), output_shape=(10, 5))
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
```
Here:
- **Input Shape**: The image input size.
- **Output Shape**: The number of categories.
- **Training**: The model is trained on the training dataset and validated on the test dataset.

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



Of course! Here is the detailed explanation in English for the `README.md` file:

```markdown
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
- **Handling Missing Values**: Utilizes imputation methods to replace missing data points, ensuring the dataset is complete and accurate.
- **Categorical Data Handling**: Converts categorical features into numerical values through various encoding techniques, making the data suitable for machine learning algorithms.
- **Data Splitting**: Divides the dataset into training and testing subsets, which is essential for evaluating the model's performance and generalization capability. This step helps in creating a robust model by preventing overfitting.
