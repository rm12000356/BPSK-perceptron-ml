import numpy as np
from config import LEARNING_RATE, NUM_EPOCHS, NUM_OF_POINTS
from data_utils import labels_insertion, data_separation

def activation_function_perceptron(y,label_type="symmetric"):
    """
    Apply perceptron activation function based on label type.
    
    Args:
        y (float): Weighted sum of inputs.
        label_type (str): 'symmetric' ([1, -1]) or 'nonsymmetric' ([0, 1]).
    
    Returns:
        int: Predicted label (1 or -1 for symmetric, 1 or 0 for nonsymmetric).
    """
    if label_type == "symmetric":
        return 1 if y >= 0 else -1
    else:
        return 1 if y >= 0 else 0  

def test_training(testing_data , weights , target , label_type="symmetric"):
    """
    Test perceptron on data and return error count and error indices.
    
    Args:
        testing_data (np.ndarray): Complex-valued test data.
        weights (np.ndarray): Perceptron weights.
        target (list): True labels.
        label_type (str): 'symmetric' or 'nonsymmetric'.
    
    Returns:
        tuple: (error_count, error_indices) where error_count is the number of
               misclassifications and error_indices is a binary array.
    """
    
    if len(testing_data) != len(target) :
        raise ValueError("Data and target lengths must match")
        
    testing_data=data_separation(testing_data)
    errors = np.zeros(len(target))
    error_count = 0

    for i, x in enumerate(testing_data):
        v = np.dot(weights, x)
        y = activation_function_perceptron(v,label_type)
        if(target[i] != y):
            error_count= error_count+1
            errors[i]=1
    return error_count,errors

def train_perceptron(data, label_type="symmetric", weights=None):
    """
    Train a perceptron on BPSK data.
    
    Args:
        data (np.ndarray): Complex-valued training data.
        label_type (str): 'symmetric' ([1, -1]) or 'nonsymmetric' ([0, 1]).
        weights (np.ndarray, optional): Initial weights. Defaults to 0.
        note about the weights:
        you can randomise the weights, but i would not recomemded it after trying and seeing its effects is not worth it
        (because there are infinite solutions  and the center of all those oppible solutionsis 0,0 so if you change it you will increment the error big time)
    Returns:
        np.ndarray: Trained weights.
    """
    target = labels_insertion(NUM_OF_POINTS, data, label_type)
    new_data = data_separation(data)
    weights = np.zeros(2) if weights is None else weights
    for i in range(NUM_EPOCHS):
        old_weights = weights.copy()
        error_count = 0
        for x, t in zip(new_data, target):
            v = np.dot(weights, x)
            y = activation_function_perceptron(v, label_type)
            weights += LEARNING_RATE * (t - y) * x
            if t != y:
                error_count += 1
        if i % 100 == 0:
            print(f"Epoch {i}, errors: {error_count}")
        if np.all(np.abs(weights - old_weights) < 1e-20):
            print(f"Converged at epoch {i+1}")
            if NUM_EPOCHS>1:
                break
    print("Final weights:", weights)
    return weights