import numpy as np
from cnn_methods import convolution, max_pool, flatten, dense, categorical_crossentropy, softmax

# Define the model
flat_size = 56 * 56 * 64  # Size after flattening the second pooling layer

# Initialize weights with improved initializers
def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def xavier_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])

def CNN_model(learning_rate, num_epochs=10, max_batches_per_epoch=20, min_delta=0.001, tolerance=0.001):
    W1 = he_initialization((3, 3, 3, 32))
    b1 = np.zeros(32)
    W2 = he_initialization((3, 3, 32, 64))
    b2 = np.zeros(64)
    W3 = xavier_initialization((flat_size, 128))
    b3 = np.zeros(128)
    W4 = xavier_initialization((128, 5))
    b4 = np.zeros(5)

    # Define the training data generator
    def simple_train_generator(X_train, y_train, batch_size):
        '''Generate batches of training data.'''
        num_samples = X_train.shape[0]
        while True:
            for offset in range(0, num_samples, batch_size):
                X_batch = X_train[offset:offset + batch_size]
                y_batch = y_train[offset:offset + batch_size]
                yield X_batch, y_batch

    # Generate synthetic training data
    X_train = np.random.rand(200, 224, 224, 3)  # 200 images of size 224x224 with 3 channels
    y_train = np.eye(5)[np.random.randint(0, 5, 200)]  # One-hot encoded labels for 5 classes
    batch_size = 32
    train_generator = simple_train_generator(X_train, y_train, batch_size)

    # Adam optimizer functions
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
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
        for X_batch, y_batch in train_generator:
            batch_count += 1
            if batch_count > max_batches_per_epoch:  # Check if max batches per epoch is reached
                break
            t += 1

            print(f"Processing batch {batch_count}")

            # Forward pass
            print("Convolution")
            conv1 = convolution(X_batch, W1, b1, stride=1, padding=1)
            conv1 = np.maximum(0, conv1)
            print("Max pooling")
            pool1 = max_pool(conv1, size=2, stride=2)
            print("Convolution")
            conv2 = convolution(pool1, W2, b2, stride=1, padding=1)
            conv2 = np.maximum(0, conv2)

            print("Max pooling")
            pool2 = max_pool(conv2, size=2, stride=2)
            print("Flatten")
            flat = flatten(pool2)
            print("Dense")
            dense1 = dense(flat, W3, b3, activation='relu')
            output = dense(dense1, W4, b4, activation='softmax')

            # Calculate loss
            loss = categorical_crossentropy(y_batch, output)
            epoch_loss += loss

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
        print(f"Epoch {epoch+1} completed, average loss: {average_loss}\n")

        # Early stopping check
        if best_loss - average_loss > min_delta:
            best_loss = average_loss
        elif average_loss - best_loss < tolerance:
            print(f"Stopping early at epoch {epoch+1} due to no significant improvement in loss.")
            break

# Running the CNN model
CNN_model(learning_rate=0.5)