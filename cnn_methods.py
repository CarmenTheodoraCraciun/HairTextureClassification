import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(X, W, b, stride=1, padding=0):
    '''Apply the 2D convolution operation.'''
    # Apply padding to the input image
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')  # Batch padding
    # Calculate the output height and width
    H_out = (X_padded.shape[1] - W.shape[0]) // stride + 1
    W_out = (X_padded.shape[2] - W.shape[1]) // stride + 1
    # Initialize the output tensor
    Y = np.zeros((X.shape[0], H_out, W_out, W.shape[3]))
    
    for n in range(X.shape[0]):  # Iterate over the batch
        for k in range(W.shape[3]):  # Iterate over the output channels
            # Convolution operation
            for i in range(0, H_out):
                for j in range(0, W_out):
                    # Extract the patch of the input image
                    patch = X_padded[n, i*stride:i*stride + W.shape[0], j*stride:j*stride + W.shape[1], :]
                    Y[n, i, j, k] = np.sum(patch * W[:, :, :, k]) + b[k]
    return Y

def max_pool(X, size=2, stride=2):
    '''Apply the max pooling operation.'''
    H_out = (X.shape[1] - size) // stride + 1
    W_out = (X.shape[2] - size) // stride + 1
    Y = np.zeros((X.shape[0], H_out, W_out, X.shape[3]))
    
    for n in range(X.shape[0]):  # Iterate over the batch
        for i in range(H_out):  # Iterate over the height
            for j in range(W_out):  # Iterate over the width
                for k in range(X.shape[3]):  # Iterate over the channels
                    Y[n, i, j, k] = np.max(X[n, i*stride:i*stride + size, j*stride:j*stride + size, k])
    return Y

def flatten(X):
    '''Flatten the input tensor into a 1D vector.'''
    return X.reshape(X.shape[0], -1)  # Batch preserving flatten

def dense(X, W, b, activation='relu'):
    '''Apply a dense layer (fully connected).'''
    Z = np.dot(X, W) + b
    if activation == 'relu':
        return np.maximum(0, Z)
    elif activation == 'softmax':
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Softmax over batches
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)
    else:
        return Z

def categorical_crossentropy(y_true, y_pred):
    '''Calculate the categorical cross-entropy loss.'''
    return -np.sum(y_true * np.log(y_pred + 1e-7)) / y_true.shape[0]  # Normalize over batch size

def softmax(Z):
    '''Apply the softmax function to the input tensor.'''
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Softmax over batches
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def check():
    image_path = './processData/curly/00cbad1ffe22d900018e5a2e7376daed4.png'
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    input_image = np.expand_dims(img, axis=0) 
    input_image = input_image.astype(np.float32) / 255.0

    kernel = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]], 
                        [[1, 0, -1], [1, 0, -1], [1, 0, -1]], 
                        [[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=np.float32)
    kernel = np.transpose(kernel, (2, 3, 1, 0))

    bias = np.array([0], dtype=np.float32)

    # Apply convolution
    convolved_output = convolution(input_image, kernel, bias, stride=1, padding=1)
    convolved_image = np.squeeze(convolved_output, axis=0) 
    convolved_image = (convolved_image - np.min(convolved_image)) / (np.max(convolved_image) - np.min(convolved_image))
    convolved_image = (convolved_image * 255).astype(np.uint8)

    # Apply max pooling
    pooled_output = max_pool(convolved_output, size=2, stride=2)
    pooled_image = np.squeeze(pooled_output, axis=0)
    pooled_image = (pooled_image - np.min(pooled_image)) / (np.max(pooled_image) - np.min(pooled_image))
    pooled_image = (pooled_image * 255).astype(np.uint8)

    # Flatten the pooled output
    flattened_output = flatten(pooled_output)
    
    # Apply dense layer
    W_dense = np.random.randn(flattened_output.shape[1], 10)
    b_dense = np.random.randn(10)
    dense_output = dense(flattened_output, W_dense, b_dense, activation='relu')

    # Apply softmax layer
    softmax_output = softmax(dense_output)

    # Calculate loss
    y_true = np.zeros((1, 10))
    y_true[0, 0] = 1
    loss = categorical_crossentropy(y_true, softmax_output)

    print("Flattened Output Shape:", flattened_output.shape)
    print("Dense Output Shape:", dense_output.shape)
    print("Softmax Output Shape:", softmax_output.shape)
    print("Categorical Crossentropy Loss:", loss)

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Convolved Image')
    plt.imshow(convolved_image)

    plt.subplot(1, 3, 3)
    plt.title('Pooled Image')
    plt.imshow(pooled_image)

    plt.show()

# check()