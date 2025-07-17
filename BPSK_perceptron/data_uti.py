from config import num_of_points
from config import num_symbols
import matplotlib.pyplot as plt
import numpy as np
# global variables that need to be always the same 
n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
x_int = np.random.randint(0, num_of_points, num_symbols)

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

def data_separation(data):
    data = np.array(data)
    separated = np.stack((np.real(data), np.imag(data)), axis=1)
    return separated

def create_data_set_normal( phase_noise = 0 , noise_power = 0):
    #np.random.seed(4)
    x_degrees = (x_int*360/num_of_points +  0)#np.random.randint(0, 360)) % 360# 45, 135, 225, 315 degree
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = (np.cos(x_radians) + 1j*np.sin(x_radians))* np.exp(1j*phase_noise) + n * np.sqrt(noise_power)
    return x_symbols
