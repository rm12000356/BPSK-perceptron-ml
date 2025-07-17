from config import learning_rate
from config import num_epochs
from config import num_of_points
from config import hidden_dim
from config import input_dim
from config import learning_rate
from config import a
from config import b
from data_uti import labels_insetion
from data_uti import trasform_labels_to_binary
from data_uti import sigmoid
from data_uti import data_separation
from data_uti import addaptability_of_output
import matplotlib.pyplot as plt
import numpy as np

def training_multilayer(Data,Target,Hlayers=1):
    output_dim = addaptability_of_output(Target)
    hidden_neuron_weights = np.random.uniform(-0.1, 0.1, size=(input_dim+1, hidden_dim))
    output_neuron_weights = np.random.uniform(-0.1, 0.1, size=(hidden_dim+1, output_dim))
    binary_target = []
    for x in range(len(Target)):
        bits = trasform_labels_to_binary(Target[x], output_dim)
        binary_target.append([int(b) for b in bits])  # make sure bits are integers
    
    binary_target = np.array(binary_target)
    Data_with_bias = data_separation(Data,1)

    for i in range(num_epochs):
        for x, d in zip(Data_with_bias, binary_target):
            # forward steps
            # step 1: calculate y
            y = np.zeros(hidden_dim+1)
            y[0]=1
            for j in range(hidden_dim):
                v = hidden_neuron_weights[:, j].T @ x
                y[j+1] = a * np.tanh(b * v)
            # step 2: calculate outputs
            o = np.zeros(output_dim)
            for j in range(output_dim):
                v = output_neuron_weights[:, j].T @ y
                o[j] = sigmoid(v)
            # step 3: calculat errors
            e = np.zeros(output_dim)
            for j in range(output_dim):
                e[j] = d[j] - o[j]
            # step 4: calculate local gradients on output layer
            lgo = np.zeros(output_dim)
            for j in range(output_dim):
                lgo[j] = e[j] * o[j] * (1 - o[j])
            # step 5: calculate local gradients on hidden layer
            lgh = np.zeros(hidden_dim)
            for j in range(hidden_dim):
                tmp_lg = 0
                for k in range(output_dim):
                    tmp_lg += lgo[k] * output_neuron_weights[j + 1, k]  
                lgh[j] = (b / a) * (a - y[j + 1]) * (a + y[j + 1]) * tmp_lg
            # step 6: calculate weights on output layer
            for j in range(hidden_dim+1):
                for k in range(output_dim):
                    output_neuron_weights[j, k] = output_neuron_weights[j, k] + learning_rate * lgo[k] * y[j]
            # step 7: calculate weights on hidden layer
            for j in range(input_dim+1):
                for k in range(hidden_dim):
                    hidden_neuron_weights[j, k] = hidden_neuron_weights[j, k] + learning_rate * lgh[k] * x[j]
    
    return hidden_neuron_weights, output_neuron_weights, binary_target

def testing( testing_data, hidden_neuron_weights,output_neuron_weights,target,rounding=1) :
    output_dim= output_neuron_weights.shape[1]
    testing_data = data_separation(testing_data,1)
    mse = 0
    for i, x in enumerate(testing_data):
        # step 1: calculate y
        y = np.zeros(hidden_dim+1)
        y[0]=1
        for j in range(hidden_dim):
            v = hidden_neuron_weights[:, j].T @ x
            y[j+1] = a * np.tanh(b * v)
        
        # step 2: calculate outputs
        if rounding==1:
            o = np.zeros(output_dim)
            for j in range(output_dim):
                v = output_neuron_weights[:, j].T @ y
                o[j] = round(sigmoid(v))
                e = 0.5*(target[i, j]-o[j])*(target[i, j]-o[j])
                mse += e
        else:
            o = np.zeros(output_dim)
            for j in range(output_dim):
                v = output_neuron_weights[:, j].T @ y
                o[j] = sigmoid(v)
                e = 0.5*(target[i, j]-o[j])*(target[i, j]-o[j])
                mse += e
    #    print( o, " the actual value is :",binary_target[i])
    
    return mse