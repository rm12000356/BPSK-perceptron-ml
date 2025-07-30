"""
Multilayer Perceptron model for 8-PSK signal classification.
"""
import numpy as np
from config import LEARNING_RATE, NUM_EPOCHS, a, b, HIDDEN_DIMS, INPUT_DIM, NUM_HIDDEN_LAYERS
from data_utils import transform_labels_to_binary, sigmoid, data_separation, adaptability_of_output

def training_multilayer(Data, Target, mse_threshold=0.01):
    """
    Train a multilayer perceptron for 8-PSK signal classification with multiple hidden layers.
    
    Args:
        Data (np.ndarray): Input data (complex or real/imag parts).
        Target (np.ndarray): Target labels for classification.
        mse_threshold (float): Stop training if MSE falls below this (default: 0.01).
    
    Returns:
        list: List of weight matrices for each layer.
        np.ndarray: Binary-encoded target labels.
    """
    output_dim = adaptability_of_output(Target)
    binary_target = np.array([transform_labels_to_binary(t, output_dim) for t in Target], dtype=int)
    Data_with_bias = data_separation(Data, 1)
    
    # Initialize weights
    weights = []
    weights.append(np.random.uniform(-0.1, 0.1, size=(INPUT_DIM+1, HIDDEN_DIMS[0])))
    for i in range(NUM_HIDDEN_LAYERS - 1):
        weights.append(np.random.uniform(-0.1, 0.1, size=(HIDDEN_DIMS[i]+1, HIDDEN_DIMS[i+1])))
    weights.append(np.random.uniform(-0.1, 0.1, size=(HIDDEN_DIMS[-1]+1, output_dim)))
    
    for epoch in range(NUM_EPOCHS):
        mse = 0
        for x, d in zip(Data_with_bias, binary_target):
            # Forward pass
            activations = [x]
            for layer in range(NUM_HIDDEN_LAYERS):
                y = np.zeros(HIDDEN_DIMS[layer] + 1)
                y[0] = 1
                for j in range(HIDDEN_DIMS[layer]):
                    v = weights[layer][:, j].T @ activations[-1]
                    y[j+1] = a * np.tanh(b * v)
                activations.append(y)
            
            o = np.zeros(output_dim)
            for j in range(output_dim):
                v = weights[-1][:, j].T @ activations[-1]
                o[j] = sigmoid(v)
            
            # Compute error and MSE
            e = d - o
            mse += np.sum(e ** 2) / output_dim
            
            # Backward pass
            gradients = [e * o * (1 - o)]
            for layer in range(NUM_HIDDEN_LAYERS - 1, -1, -1):
                next_gradients = np.zeros(HIDDEN_DIMS[layer])
                for j in range(HIDDEN_DIMS[layer]):
                    tmp_lg = np.sum(gradients[0] * weights[layer + 1][j + 1, :])
                    next_gradients[j] = (b / a) * (a - activations[layer + 1][j + 1]) * (a + activations[layer + 1][j + 1]) * tmp_lg
                gradients.insert(0, next_gradients)
            
            # Update weights
            for layer in range(NUM_HIDDEN_LAYERS + 1):
                for j in range(weights[layer].shape[0]):
                    for k in range(weights[layer].shape[1]):
                        weights[layer][j, k] += LEARNING_RATE * gradients[layer][k] * activations[layer][j]
        
        mse /= len(Data_with_bias)
        if mse < mse_threshold:
            print(f"Stopping early at epoch {epoch+1} with MSE {mse:.4f}")
            break
    
    return weights, binary_target

def testing(testing_data, weights, target, rounding=1):
    """
    Test the multilayer perceptron on input data.
    
    Args:
        testing_data (np.ndarray): Input data for testing (complex or real/imag parts).
        weights (list): List of weight matrices for each layer.
        target (np.ndarray): Binary-encoded target labels.
        rounding (int): 1 to round outputs, 0 for raw sigmoid outputs (default: 1).
    
    Returns:
        tuple: (Mean squared error, accuracy) over the test set.
    """
    output_dim = weights[-1].shape[1]
    testing_data = data_separation(testing_data, 1)
    mse = 0
    correct = 0
    for i, x in enumerate(testing_data):
        # Forward pass
        activations = [x]
        for layer in range(NUM_HIDDEN_LAYERS):
            y = np.zeros(HIDDEN_DIMS[layer] + 1)
            y[0] = 1
            for j in range(HIDDEN_DIMS[layer]):
                v = weights[layer][:, j].T @ activations[-1]
                y[j+1] = a * np.tanh(b * v)
            activations.append(y)
        
        o = np.zeros(output_dim)
        for j in range(output_dim):
            v = weights[-1][:, j].T @ activations[-1]
            o[j] = round(sigmoid(v)) if rounding == 1 else sigmoid(v)
            e = 0.5 * (target[i, j] - o[j]) ** 2
            mse += e
        
        # Compute accuracy
        predicted = np.round(o).astype(int)
        if np.array_equal(predicted, target[i]):
            correct += 1
    
    mse /= len(testing_data)
    accuracy = correct / len(testing_data)
    return mse, accuracy