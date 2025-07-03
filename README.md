# Hair Texture Classification

This project explores different machine learning models for classifying hair textures into five categories: **curly**, **straight**, **wavy**, **dreadlocks**, and **kinky**. The goal is to evaluate and compare models ranging from traditional approaches to deep learning methods, with a particular focus on **classification accuracy and robustness**.

## 📦 Dataset

The project uses the **[Hair Type Dataset](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset)** from Kaggle, which includes labeled images in the five hair texture categories. This data can be found in the `\originalData` folder.

## ✅ Evaluation Metrics

Each model was evaluated using key classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

These metrics help compare models not only by general performance, but also by their **robustness in identifying visually similar hair textures** (e.g., dreadlocks vs. kinky). 

## 📁 Scripts

* `DataProcessing_HairTexture.ipynb`
  → Handles all preprocessing steps including image resizing, normalization, and data augmentation. Also visualizes class distribution and prepares the dataset for training. There is not need for this script to be run all the time, the data are all save in the folder `\processData`.

* `SLP_HairTexture.ipynb`
  → Implements a **Single-Layer Perceptron** architecture using EfficientNetB0 as a feature extractor, followed by a global average pooling layer and fully connected classifier. It achieved the best accuracy among all models.

* `MLP_HairTexture.ipynb`
  → Implements a **Multi-Layer Perceptron** based on the SLP structure, but with additional hidden layers and regularization (Dropout + L2). Shows strong classification performance, particularly on the dreadlocks class.

* `CNN_HairTexture.ipynb`
  → **(In Progress)** – Contains a custom **Convolutional Neural Network** designed to learn spatial features from images. Incorporates techniques such as BatchNormalization, Dropout, and EarlyStopping for training.

* `FNN_HairTexture.ipynb`
  → **(In Progress)** – Implements a basic **Feedforward Neural Network** using flattened image input. Current version has limited performance; improvements like adding hidden layers and regularization are planned.

* `otherModels_HairTexture.ipynb`
  → **(In Progress)** – Contains traditional machine learning models for comparison:

  * **kNN** – A distance-based model using cosine similarity and PCA for dimensionality reduction. Performs modestly due to limitations in visual feature extraction.
  * **Random Forest** – Uses an ensemble of decision trees for classification. Performs better than kNN but struggles with visually similar classes.
  * **XGBoost** – Gradient boosting method with better generalization than Random Forest. Applies PCA to improve class separation, but still underperforms compared to neural models.

## 📊 Current Status

| Model         | Status         | Accuracy   |
| ------------- | -------------- | ---------- |
| SLP           | ✅ Completed    | **96.96%** |
| MLP           | ✅ Completed    | **90.99%** |
| CNN           | 🚧 In Progress | 80.24%     |
| FNN           | 🚧 In Progress | 31.16%     |
| XGBoost       | 🚧 In Progress | 46.03%     |
| Random Forest | 🚧 In Progress | 43.88%     |
| KNN           | 🚧 In Progress | 39.56%     |

## 📈 Key Insights

* **SLP** achieved the **highest performance**, proving that fully connected architectures with strong preprocessing and regularization can outperform even CNNs in some visual classification tasks.
* **CNN** excelled at recognizing spatial features, particularly for curly and dreadlocks classes.
* **MLP** offered a balance between simplicity and power, with robust classification across all categories.
* Traditional models like **kNN**, **Random Forest**, and **XGBoost** struggled with texture nuances despite dimensionality reduction via PCA.
* **FNN** had the weakest performance, showing the importance of deeper architectures and regularization.

## 🧪 Technologies Used

| Category      | Tools / Libraries                        |
| ------------- | ---------------------------------------- |
| Language      | Python                                   |
| ML Frameworks | TensorFlow, Keras, scikit-learn, XGBoost |
| Data Handling | NumPy, Pandas, OpenCV                    |
| Visualization | Matplotlib, Seaborn                      |
| Environment   | Jupyter Notebook                         |
