## This file contain

import os
import cv2
import numpy as np
import random
import shutil

def is_image(file_path):
    try:
        img = cv2.imread(file_path)
        return img is not None
    except:
        return False

def resize_image(img, size):
    '''Resize an image using bilinear interpolation.'''
    # Get the original dimensions of the image
    original_height, original_width, _ = img.shape
    new_width, new_height = size
    # Create an empty image with the new dimensions
    resized_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_width):
        for j in range(new_height):
            # Calculate the corresponding position in the original image
            x = i * (original_width - 1) / (new_width - 1)
            y = j * (original_height - 1) / (new_height - 1)
            
            # Find the surrounding pixels
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, original_width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, original_height - 1)
            
            # Get pixel values from the original image
            Ia = img[y0, x0]
            Ib = img[y0, x1]
            Ic = img[y1, x0]
            Id = img[y1, x1]
            
            # Compute the weights for each pixel
            wa = (x1 - x) * (y1 - y)
            wb = (x - x0) * (y1 - y)
            wc = (x1 - x) * (y - y0)
            wd = (x - x0) * (y - y0)
            
            # Calculate the new pixel value
            pixel = wa * Ia + wb * Ib + wc * Ic + wd * Id
            # Assign the new pixel value to the resized image
            resized_img[j, i] = np.round(pixel).astype(int)
    
    return resized_img

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    '''Converting images to the same size and possibly normalizing pixels to improve model performance.'''
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Start processing data.")
    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        # Create a directory for the category in the output directory if it doesn't exist
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
                        # Resize the image
                        img = resize_image(img, size)
                        # Save the image as a PNG
                        new_img_name = os.path.splitext(img_name)[0] + '.png'
                        cv2.imwrite(os.path.join(output_category_dir, new_img_name), img)
                    else:
                        print(f"Failed to load image: {img_path}")
                else:
                    print(f"Not an image: {img_path}")
    print("The data are processed.")

preprocess_images('./originalData', './processData')

def preprocess_and_augment_image(img, size=(224, 224), shear_range=0.1, zoom_range=0.1, horizontal_flip=True):
    '''Preprocess and augment an image by applying resizing, shearing, zooming, and horizontal rotation.'''
    
    # Shearing
    rows, cols, ch = img.shape
    shear_factor = np.random.uniform(-shear_range, shear_range)
    M_shear = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    img = cv2.warpAffine(img, M_shear, (cols, rows))

    # Zooming
    zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

    # Ensuring original dimensions after zooming
    if zoom_factor < 1:
        new_height, new_width = img.shape[:2]
        pad_height = (rows - new_height) // 2
        pad_width = (cols - new_width) // 2
        img = cv2.copyMakeBorder(img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        start_x = (img.shape[1] - cols) // 2
        start_y = (img.shape[0] - rows) // 2
        img = img[start_y:start_y + rows, start_x:start_x + cols]

    # Horizontal flip
    if horizontal_flip and np.random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Resizing
    img = cv2.resize(img, size)

    return img

def create_validation_set(input_dir, output_dir, split_ratio=0.2):
    '''Creates a validation dataset by copying a percentage of existing data.'''
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Start getting data for validation.")
    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        # Create a directory for the category in the output directory if it doesn't exist
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        if os.path.isdir(category_dir):
            # Get a list of image files in the category directory
            images = [f for f in os.listdir(category_dir) if is_image(os.path.join(category_dir, f))]
            random.shuffle(images)
            
            # Determine the number of images to use for validation
            split_index = int(len(images) * split_ratio)
            validation_images = images[:split_index]

            for img_name in validation_images:
                img_path = os.path.join(category_dir, img_name)
                if is_image(img_path):
                    img = cv2.imread(img_path)
                    # Preprocess and augment the image
                    augmented_img = preprocess_and_augment_image(img)
                    # Save the augmented image
                    new_img_name = os.path.splitext(img_name)[0] + '_aug.png'
                    cv2.imwrite(os.path.join(output_category_dir, new_img_name), augmented_img)
                else:
                    # Copy non-image files as is
                    shutil.copy(img_path, os.path.join(output_category_dir, img_name))

            num_images = len(validation_images)
            print(f"Folder validationData/{category} has {num_images} images.")

    print("Validation data set created.")

create_validation_set('./processData', './validationData', split_ratio=0.2)