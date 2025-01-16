import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Input
from keras.models import Model

# Set environment variables for parallelism
os.environ["OMP_NUM_THREADS"] = "4"  # Number of OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Number of MKL threads
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Optimized Dataset Generators
def simple_train_generator(X_train, y_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def simple_val_generator(X_val, y_val, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Weight Initialization Functions
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def xavier_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])

# Define the CNN Model with Distributed Strategy
strategy = tf.distribute.MirroredStrategy()
# Initialize lists for tracking losses
train_losses = []
val_losses = []

with strategy.scope():
    def CNN_model(learning_rate, num_epochs=10, max_batches_per_epoch=20, min_delta=0.001, tolerance=0.001):
        best_loss = float('inf')

        input_layer = Input(shape=(224, 224, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        
        flat_output = tf.keras.layers.Flatten()(pool1)
        flat_size = flat_output.shape[1]

        W3 = xavier_initialization((flat_size, 128))
        b3 = np.zeros(128)
        W4 = xavier_initialization((128, 5))
        b4 = np.zeros(5)

        X_train = np.random.rand(200, 224, 224, 3)
        y_train = np.eye(5)[np.random.randint(0, 5, 200)]
        X_val = np.random.rand(50, 224, 224, 3)
        y_val = np.eye(5)[np.random.randint(0, 5, 50)]

        batch_size = 32
        train_dataset = simple_train_generator(X_train, y_train, batch_size)
        val_dataset = simple_val_generator(X_val, y_val, batch_size)

        conv_model = Model(inputs=input_layer, outputs=pool1)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            batch_count = 0

            for X_batch, y_batch in train_dataset.take(max_batches_per_epoch):
                batch_count += 1

                conv_output = conv_model.predict(X_batch)
                flat = np.reshape(conv_output, (conv_output.shape[0], -1))
                dense1 = np.maximum(0, np.dot(flat, W3) + b3)
                output = tf.nn.softmax(np.dot(dense1, W4) + b4).numpy()

                loss = -np.sum(y_batch * np.log(output + 1e-8)) / batch_size
                epoch_loss += loss

                accuracy = np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1)) / batch_size
                epoch_accuracy += accuracy

                grad_output = output - y_batch
                grad_W4 = np.dot(dense1.T, grad_output)
                grad_b4 = np.sum(grad_output, axis=0)
                grad_dense1 = np.dot(grad_output, W4.T) * (dense1 > 0)
                grad_W3 = np.dot(flat.T, grad_dense1)
                grad_b3 = np.sum(grad_dense1, axis=0)

                W4 -= learning_rate * grad_W4
                b4 -= learning_rate * grad_b4
                W3 -= learning_rate * grad_W3
                b3 -= learning_rate * grad_b3

            val_loss = 0
            val_accuracy = 0
            for X_batch, y_batch in val_dataset.take(max_batches_per_epoch):
                conv_output = conv_model.predict(X_batch)
                flat = np.reshape(conv_output, (conv_output.shape[0], -1))
                dense1 = np.maximum(0, np.dot(flat, W3) + b3)
                output = tf.nn.softmax(np.dot(dense1, W4) + b4).numpy()
                loss = -np.sum(y_batch * np.log(output + 1e-8)) / batch_size
                val_loss += loss

                accuracy = np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1)) / batch_size
                val_accuracy += accuracy

            train_losses.append(epoch_loss / batch_count)
            train_accuracies.append(epoch_accuracy / batch_count)
            val_losses.append(val_loss / max_batches_per_epoch)
            val_accuracies.append(val_accuracy / max_batches_per_epoch)

            print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

        print(f"\nFinal Train Accuracy: {train_accuracies[-1]:.4f}, Final Val Accuracy: {val_accuracies[-1]:.4f}")
        return conv_model

# Train the Model
conv_model = CNN_model(learning_rate=0.001, num_epochs=50, max_batches_per_epoch=70)

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()