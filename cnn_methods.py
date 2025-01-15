import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(X, W, b, stride=1, padding=0):
  """
  Applies a 2D convolution operation.

  Args:
      X: Input image.
      W: Filter (kernel).
      b: Bias term.
      stride: Stride of the convolution.
      padding: Amount of padding to apply.

  Returns:
      The convolved output.
  """
  # Apply padding to the input image
  X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')  # Padding for batches
  # Calculate output height and width
  H_out = (X_padded.shape[1] - W.shape[0]) // stride + 1
  W_out = (X_padded.shape[2] - W.shape[1]) // stride + 1
  # Initialize output tensor
  Y = np.zeros((X.shape[0], H_out, W_out, W.shape[3]))

  for n in range(X.shape[0]):  # Iterate over batches
    for k in range(W.shape[3]):  # Iterate over output channels
      # Convolution operation
      for i in range(0, H_out):
        for j in range(0, W_out):
          # Extract patch from input image
          patch = X_padded[n, i*stride:i*stride + W.shape[0], j*stride:j*stride + W.shape[1], :]
          Y[n, i, j, k] = np.sum(patch * W[:, :, :, k]) + b[k]
  return Y

def max_pool(X, size=2, stride=2):
  """
  Applies max pooling operation.

  Args:
      X: Input image.
      size: Pooling window size.
      stride: Stride of the pooling.

  Returns:
      The max pooled output.
  """
  H_out = (X.shape[1] - size) // stride + 1
  W_out = (X.shape[2] - size) // stride + 1
  Y = np.zeros((X.shape[0], H_out, W_out, X.shape[3]))

  for n in range(X.shape[0]):  # Iterate over batches
    for i in range(H_out):  # Iterate over height
      for j in range(W_out):  # Iterate over width
        for k in range(X.shape[3]):  # Iterate over channels
          Y[n, i, j, k] = np.max(X[n, i*stride:i*stride + size, j*stride:j*stride + size, k])
  return Y

def flatten(X):
  """
  Flattens an input tensor into a 1D vector.

  Args:
      X: Input tensor.

  Returns:
      The flattened vector.
  """
  return X.reshape(X.shape[0], -1)  # Flatten preserving batch

def dense(X, W, b, activation='relu'):
  """
  Applies a dense (fully connected) layer.

  Args:
      X: Input data.
      W: Weights of the layer.
      b: Biases of the layer.
      activation: Activation function to apply (default: relu).

  Returns:
      The output of the dense layer.
  """
  Z = np.dot(X, W) + b
  if activation == 'relu':
    return np.maximum(0, Z)
  elif activation == 'softmax':
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Softmax across batches
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)
  else:
    return Z

def categorical_crossentropy(y_true, y_pred):
  """
  Calculates the categorical cross-entropy loss.

  Args:
      y_true: True labels.
      y_pred: Predicted labels.

  Returns:
      The categorical cross-entropy loss.
  """
  return -np.sum(y_true * np.log(y_pred + 1e-7)) / y_true.shape[0]  #