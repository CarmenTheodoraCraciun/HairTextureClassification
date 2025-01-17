import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

def load_images_and_labels(input_dir, size=(128, 128)):
    images = []
    labels = []
    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        if os.path.isdir(category_dir):
            for img_name in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, size)
                    images.append(img)
                    labels.append(category)
    return np.array(images), np.array(labels)

input_dir = './processData'
images, labels = load_images_and_labels(input_dir)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the images
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

# Convert labels to numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode the labels
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Augmentare date
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare training data
train_datagen = datagen.flow(X_train_normalized, y_train_categorical, batch_size=32)
validation_datagen = ImageDataGenerator().flow(X_test_normalized, y_test_categorical, batch_size=32)

# Loading the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential()
model.add(base_model) # The VGG16 model
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_datagen, epochs=20, validation_data=validation_datagen)

# Evaluează modelul
loss, accuracy = model.evaluate(X_test_normalized, y_test_categorical)
print(f'Test accuracy: {accuracy:.4f}')

# Graficul pentru accuracy și val_accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, 'b', label='Accuracy on training data')
plt.plot(epochs, val_accuracy, 'r', label='Accuracy on validation data')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictii pentru setul de test
y_pred_encoded = model.predict(X_test_normalized)
y_pred = np.argmax(y_pred_encoded, axis=1)
y_test = np.argmax(y_test_categorical, axis=1)

# Calculează precizia, recall și F1-score
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matricea de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)

# Afișează matricea de confuzie
plt.figure(figsize=(10, 7))
sns.heatmap(conf_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confuzion matrix")
plt.ylabel("Real classes")
plt.xlabel("Predicted classes")
plt.show()