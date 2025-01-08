print("Prepocess image")

import os
import cv2
import shutil
import random

def is_image(file_path):
    try:
        img = cv2.imread(file_path)
        return img is not None
    except:
        return False

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    '''Converting images to the same size and perhaps normalizing pixels to improve model performance.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Start processing data.")
    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        # Check if category_dir is a directory
        if os.path.isdir(category_dir):
            for img_name in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_name)
                
                # Check if the file is an image
                if is_image(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, size)
                        # Save image as PNG
                        new_img_name = os.path.splitext(img_name)[0] + '.png'
                        cv2.imwrite(os.path.join(output_category_dir, new_img_name), img)
                    else:
                        print(f"Failed to load image: {img_path}")
                else:
                    print(f"Not an image: {img_path}")
    print("The data are process.")

# preprocess_images('./originalData', './processData')
print("\n-----------------------\n")

print("Create validation set")

# Function to create validation set
def create_validation_set(input_dir, output_dir, split_ratio=0.2):
    '''Creează un set de date de validare, copiind un procentaj din datele existente.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        # Check if category_dir is a directory
        if os.path.isdir(category_dir):
            images = [f for f in os.listdir(category_dir) if is_image(os.path.join(category_dir, f))]
            random.shuffle(images)
            
            split_index = int(len(images) * split_ratio)
            validation_images = images[:split_index]

            for img_name in validation_images:
                img_path = os.path.join(category_dir, img_name)
                shutil.copy(img_path, os.path.join(output_category_dir, img_name))

            # Print message with the number of images in the validation category folder
            num_images = len(validation_images)
            print(f"Folder validationData/{category} has {num_images} images.")

    print("Validation data set created.")

# Create validation set
# create_validation_set('./processData', './validationData', split_ratio=0.2)

print("\n-----------------------\n")
print("Data Augmentation and Loading")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image Data Generator for training with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Image Data Generator for validation (without augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Loading the training and validation datasets
train_generator = train_datagen.flow_from_directory(
    './processData', 
    target_size=(224, 224),
    batch_size=32, 
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    './validationData', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


print("\n-----------------------\n")
print("Building the CNN Model")

# Construirea modelului
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # Ajustăm la 5 clase de păr
])

# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n-----------------------\n")
print("Model Training")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)

print("\n-----------------------\n")
print("Evaluating and Saving the Model")
# Evaluarea modelului
loss, accuracy = model.evaluate(valid_generator)
print(f'Validation Accuracy: {accuracy:.2f}')

# Salvarea modelului
model.save('hair_type_classification_model.h5')