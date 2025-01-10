from keras.layers import Conv2D, Input
from keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cnn_methods import max_pool, flatten, dense, categorical_crossentropy, softmax

# Initialize weights with improved initializers
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def xavier_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])

def simple_train_generator(X_train, y_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

def simple_val_generator(X_val, y_val, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

train_losses = []
val_losses = []

def CNN_model(learning_rate, num_epochs=10, max_batches_per_epoch=20, min_delta=0.001, tolerance=0.001):
    flat_size = 802816  # Size after flattening the second pooling layer
    W3 = xavier_initialization((flat_size, 128))
    b3 = np.zeros(128)
    W4 = xavier_initialization((128, 5))
    b4 = np.zeros(5)

    X_train = np.random.rand(200, 224, 224, 3)  # 200 images of size 224x224 with 3 channels
    y_train = np.eye(5)[np.random.randint(0, 5, 200)]  # One-hot encoded labels for 5 classes
    X_val = np.random.rand(50, 224, 224, 3)  # 50 images of size 224x224 with 3 channels
    y_val = np.eye(5)[np.random.randint(0, 5, 50)]  # One-hot encoded labels for 5 classes

    batch_size = 32
    train_dataset = simple_train_generator(X_train, y_train, batch_size)
    val_dataset = simple_val_generator(X_val, y_val, batch_size)

    # Define the model with Keras Conv2D layers
    input_layer = Input(shape=(224, 224, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv_model = Model(inputs=input_layer, outputs=conv2)

    # Adam optimizer functions
    m_W3, v_W3 = np.zeros_like(W3), np.zeros_like(W3)
    m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)
    m_W4, v_W4 = np.zeros_like(W4), np.zeros_like(W4)
    m_b4, v_b4 = np.zeros_like(b4), np.zeros_like(b4)

    def adam_update(W, dW, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        m = beta1 * m + (1 - beta1) * dW
        v = beta2 * v + (1 - beta2) * (dW ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return W, m, v

    # Training loop
    t = 0  # Timestep counter for Adam optimizer
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        batch_count = 0
        correct_predictions_train = 0
        total_predictions_train = 0

        for X_batch, y_batch in train_dataset.take(max_batches_per_epoch):
            batch_count += 1
            t += 1

            # Forward pass
            conv_output = conv_model.predict(X_batch)
            pool1 = max_pool(conv_output, size=2, stride=2)
            flat = flatten(pool1)
            dense1 = dense(flat, W3, b3, activation='relu')
            output = dense(dense1, W4, b4, activation='softmax')

            # Calculate loss
            loss = categorical_crossentropy(y_batch, output)
            epoch_loss += loss

            # Calculate accuracy
            correct_predictions_train += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
            total_predictions_train += len(y_batch)

            # Backward pass and weight update with Adam optimizer
            grad_output = output - y_batch
            grad_W4 = np.dot(dense1.T, grad_output)
            grad_b4 = np.sum(grad_output, axis=0)
            grad_dense1 = np.dot(grad_output, W4.T) * (dense1 > 0)
            grad_W3 = np.dot(flat.T, grad_dense1)
            grad_b3 = np.sum(grad_dense1, axis=0)

            W4, m_W4, v_W4 = adam_update(W4, grad_W4, m_W4, v_W4, t)
            b4, m_b4, v_b4 = adam_update(b4, grad_b4, m_b4, v_b4, t)
            W3, m_W3, v_W3 = adam_update(W3, grad_W3, m_W3, v_W3, t)
            b3, m_b3, v_b3 = adam_update(b3, grad_b3, m_b3, v_b3, t)

        average_loss = epoch_loss / batch_count
        train_losses.append(average_loss)  # Stocăm pierderile de antrenament
        accuracy_train = correct_predictions_train / total_predictions_train
        print(f"Epoch {epoch+1} completed, average loss: {average_loss}, train accuracy: {accuracy_train:.2f}")

        # Validare
        val_loss = 0
        val_batch_count = 0
        correct_predictions_val = 0
        total_predictions_val = 0

        for X_batch, y_batch in val_dataset.take(max_batches_per_epoch):
            val_batch_count += 1

            # Forward pass
            conv_output = conv_model.predict(X_batch)
            pool1 = max_pool(conv_output, size=2, stride=2)
            flat = flatten(pool1)
            dense1 = dense(flat, W3, b3, activation='relu')
            output = dense(dense1, W4, b4, activation='softmax')

            # Calculate loss
            loss = categorical_crossentropy(y_batch, output)
            val_loss += loss

            # Calculate accuracy
            correct_predictions_val += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
            total_predictions_val += len(y_batch)

        average_val_loss = val_loss / val_batch_count
        val_losses.append(average_val_loss)  # Stocăm pierderile de validare
        accuracy_val = correct_predictions_val / total_predictions_val
        print(f"Epoch {epoch+1} validation loss: {average_val_loss}, validation accuracy: {accuracy_val:.2f}\n")

        # Early stopping check
        if best_loss - average_val_loss > min_delta:
            best_loss = average_val_loss
        elif average_val_loss - best_loss < tolerance:
            print(f"\n\nStopping early at epoch {epoch+1} due to no significant improvement in validation loss.")
            break
    
    return conv_model, accuracy_train, accuracy_val

def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    return confusion_matrix

# Running the CNN model
conv_model, train_accuracy, val_accuracy = CNN_model(learning_rate=0.5, num_epochs=10, max_batches_per_epoch=20)

print(f"\nFinal Train Accuracy: {train_accuracy:.2f}")
print(f"Final Validation Accuracy: {val_accuracy:.2f}")

X_test = np.random.rand(10, 224, 224, 3)  # 10 imagini de test de dimensiune 224x224 cu 3 canale
y_test = np.eye(5)[np.random.randint(0, 5, 10)]  # Etichete codate one-hot pentru 5 clase
predictions = np.argmax(conv_model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(compute_confusion_matrix(y_test_labels, predictions, num_classes=5))    

# Vizualizarea Pierderii
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during training and validation')
plt.legend()
plt.show()