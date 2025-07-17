from config import learning_rate
from config import num_epochs
from config import num_of_points
from data_uti import labels_insetion
from data_uti import data_separation
import matplotlib.pyplot as plt
import numpy as np

def activation_function_perceptron(y):
    if y <= 0:
        return -1
    else :
        return 1

def test_training(testing_data , weights , target , error=0):
    testing_data=data_separation(testing_data)
    for i, x in enumerate(testing_data):
        v = np.dot(weights, x)
        y = activation_function_perceptron(v)
        if(target[i] != y):
           error= error+1
    return error

def train_perceptron(data,weights = np.zeros(2)):
    Target_for_given_data = labels_insetion(num_of_points,data)
    new_data = data_separation(data)
    for i in range(num_epochs):
        for x, target in zip(new_data, Target_for_given_data):
            v = np.dot(weights.T, x)
            y = activation_function_perceptron(v)
            weights = weights + learning_rate * (target - y)*x
    return weights