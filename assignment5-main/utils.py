# Submitted by: |Joseph Shepherd, joseshep| |Venkata Naga Sreya Kolachalama, vekola|

import numpy as np

def euclidean_distance(x1, x2):
    # Calculate the Euclidean distance between two vectors x1 and x2.
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

def manhattan_distance(x1, x2):
    # Calculate the Manhattan distance between two vectors x1 and x2.
    distance = np.sum(np.abs(x1 - x2))
    return distance

def identity(x, derivative=False):
    # Identity derivatives are always 1!
    if derivative:
        return np.ones_like(x)
    else:
        return x

def sigmoid(x, derivative=False):
    # Sigmoid activation function, returns sigmoid(x) through recursion or its derivative.
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
    # Hyperbolic tangent activation function, returns tanh(x) or its derivative.
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)

def relu(x, derivative=False):
    # Rectified Linear Unit (ReLU) activation function, returns max(0, x) or its derivative.
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)

def softmax(x, derivative=False):
    # Softmax activation function, returns softmax(x) or its derivative.
    x = np.clip(x, -1e100, 1e100)

    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))
    else:
        return softmax(x) * (1 - softmax(x))

def cross_entropy(y, p):
    # Calculate the cross-entropy loss between true labels y and predicted probabilities p.
    epsilon = 1e-15
    
    p = np.clip(p, epsilon, 1 - epsilon)
    loss = -np.sum(y * np.log(p)) / len(y)
    return loss

def one_hot_encoding(y):
    # Convert categorical labels in y to one-hot encoded vectors.
    unique_values = np.unique(y)
    one_hot_encoded = np.zeros((len(y), len(unique_values)))

    for i, value in enumerate(y):
        index = np.where(unique_values == value)[0][0]
        one_hot_encoded[i, index] = 1
    return one_hot_encoded