import numpy as np

def convolution(X, W, b, stride=1, padding=0):
    '''Apply the 2D convolution operation.'''
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')  # Batch padding
    H_out = (X_padded.shape[1] - W.shape[0]) // stride + 1
    W_out = (X_padded.shape[2] - W.shape[1]) // stride + 1
    Y = np.zeros((X.shape[0], H_out, W_out, W.shape[3]))
    
    for n in range(X.shape[0]):  # Iterate over the batch
        for i in range(0, H_out, stride):
            for j in range(0, W_out, stride):
                for k in range(W.shape[3]):
                    Y[n, i, j, k] = np.sum(X_padded[n, i:i + W.shape[0], j:j + W.shape[1], :] * W[:, :, :, k]) + b[k]
    
    return Y

def max_pool(X, size=2, stride=2):
    '''Apply the max pooling operation.'''
    H_out = (X.shape[1] - size) // stride + 1
    W_out = (X.shape[2] - size) // stride + 1
    Y = np.zeros((X.shape[0], H_out, W_out, X.shape[3]))
    
    for n in range(X.shape[0]):  # Iterate over the batch
        for i in range(0, H_out, stride):
            for j in range(0, W_out, stride):
                for k in range(X.shape[3]):
                    Y[n, i, j, k] = np.max(X[n, i:i + size, j:j + size, k])
    
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
    return -np.sum(y_true * np.log(y_pred + 1e-7)) / y_true.shape[0]  # Normalize over batch size

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Softmax over batches
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

# Define the model
flat_size = 56 * 56 * 64

# Initialize weights
W1 = np.random.rand(3, 3, 3, 32)
b1 = np.random.rand(32)
W2 = np.random.rand(3, 3, 32, 64)
b2 = np.random.rand(64)
W3 = np.random.rand(flat_size, 128)
b3 = np.random.rand(128)
W4 = np.random.rand(128, 5)
b4 = np.random.rand(5)

num_epochs = 10
learning_rate = 0.1
tolerance = 0.00001  # Set your tolerance for early stopping
min_delta = 0.0001  # Minimum change to be considered an improvement

# Define the training data generator
def simple_train_generator(X_train, y_train, batch_size):
    num_samples = X_train.shape[0]
    while True:
        for offset in range(0, num_samples, batch_size):
            X_batch = X_train[offset:offset + batch_size]
            y_batch = y_train[offset:offset + batch_size]
            yield X_batch, y_batch

# Example usage
X_train = np.random.rand(1000, 224, 224, 3)
y_train = np.eye(5)[np.random.randint(0, 5, 1000)]
batch_size = 32
train_generator = simple_train_generator(X_train, y_train, batch_size)

# Training loop
best_loss = float('inf')
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    batch_count = 0
    for X_batch, y_batch in train_generator:
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processing batch {batch_count}")

        # Forward pass
        print("Forward pass - first convolution layer")
        conv1 = convolution(X_batch, W1, b1, stride=1, padding=1)
        conv1 = np.maximum(0, conv1)
        
        print("Forward pass - first pooling layer")
        pool1 = max_pool(conv1, size=2, stride=2)
        
        print("Forward pass - second convolution layer")
        conv2 = convolution(pool1, W2, b2, stride=1, padding=1)
        conv2 = np.maximum(0, conv2)
        
        print("Forward pass - second pooling layer")
        pool2 = max_pool(conv2, size=2, stride=2)
        
        print("Forward pass - flattening")
        flat = flatten(pool2)
        
        print("Forward pass - first dense layer")
        dense1 = dense(flat, W3, b3, activation='relu')
        
        print("Forward pass - output dense layer")
        output = dense(dense1, W4, b4, activation='softmax')

        # Calculate loss
        loss = categorical_crossentropy(y_batch, output)
        epoch_loss += loss
        print(f"Loss for current batch: {loss}")

        # Backward pass and weight update (simplified)
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

    average_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1} completed, average loss: {average_loss}")
    
    # Check for improvement
    if best_loss - average_loss > min_delta:
        best_loss = average_loss
    elif average_loss - best_loss < tolerance:
        print(f"Stopping early at epoch {epoch+1} due to no significant improvement in loss.")
        break