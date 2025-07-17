from config import num_of_points
from config import num_symbols
import matplotlib.pyplot as plt
import numpy as np
# global variables that need to be always the same 
n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
x_int = np.random.randint(0, num_of_points, num_symbols)

# creates the labels for a data set so if we have 4 points in a constelation we have 4 labels this functinon creates those 4 labels  

def labels_creation(num_of_points):
    base=0
    possible_labels = []
    for l in range(0,num_of_points):
        if l%2 == 0:
            possible_labels.append(1 + base * 2)
        else:
            possible_labels.append(-1 - base * 2)
            base=base+1
    return possible_labels
    
#this function takes the data use the previous function to create the set of labels and given the data assigns the value of said data to a label

def labels_insetion(num_of_points , data):
    labels=labels_creation(num_of_points)
    identification_of_lables_in_data = []

    reference_symbol = []
    for x in range(num_of_points):
        x_degrees = (x*360/num_of_points +  0) 
        x_radians = x_degrees*np.pi/180.0
        reference_symbol.append(np.cos(x_radians) + 1j*np.sin(x_radians))
    
    for d in data:
        for i in range(0,num_of_points):
            if np.isclose(d.real, reference_symbol[i].real, atol=1e-20) and \
               np.isclose(d.imag, reference_symbol[i].imag, atol=1e-20):
                identification_of_lables_in_data.append(labels[i])
                break

    return identification_of_lables_in_data

#as the name says after creating the label why use x number of output for a x labels, so i tranforms the labels to a set of binary data, with the value of said 
#label and make it more efficient, i think works with -neg and +ve

def trasform_labels_to_binary(value, outputlayer):
    if value >=0:
        s = bin(value) 
        x= s[2:].zfill(outputlayer)
        return list(x)
    else:
        s = bin(value) 
        x= s[3:].zfill(outputlayer-1)
        y= '1' + x
        return list(y)
        
#automaticaly change the output depending of the minimim requirement aka the number of bits ok the max value in the labels

def addaptability_of_output(label):
    max_value_of_label= max(label)
    x= np.ceil(np.log2(max_value_of_label)) +1
    return int(x)

# when working with img numbers it a requirement to separate the img to the real, i added bias since is a requirment in some cases 

def data_separation(data , bias=0):
    if(bias == 0):
        data = np.array(data)
        separated = np.stack((np.real(data), np.imag(data)), axis=1)
    else :
        data = np.array(data)
        separated = np.stack((np.ones(len(data)), np.real(data), np.imag(data)), axis=1)
    return separated

#this is a activation function for multilayer perceptron

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

#here the data set is created 

def create_data_set_normal( phase_noise = 0 , noise_power = 0,random_angle=0):
    x_degrees = (x_int*360/num_of_points + 0)#random_angle*np.random.randint(0, 360)) % 360# 45, 135, 225, 315 degree
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = (np.cos(x_radians) + 1j*np.sin(x_radians))* np.exp(1j*phase_noise) + n * np.sqrt(noise_power)
    return x_symbols
