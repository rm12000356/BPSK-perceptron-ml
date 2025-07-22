import numpy as np
from config import NUM_OF_POINTS, NUM_SYMBOLS

def labels_creation_Symmetric(NUM_OF_POINTS):
    """
    Create symmetric labels for BPSK (e.g., [1, -1] for 2 points).
    
    Args:
        num_points (int): Number of constellation points.
    
    Returns:
        list: Symmetric labels (e.g., [1, -1] for BPSK).
    """
    base=0
    possible_labels = []
    for l in range(0,NUM_OF_POINTS):
        if l%2 == 0:
            possible_labels.append(1 + base * 2)
        else:
            possible_labels.append(-1 - base * 2)
            base=base+1
    return possible_labels

def labels_creation_NonSymmetric(NUM_OF_POINTS):
    """
    Create non-symmetric labels for BPSK (e.g., [0, 1] for 2 points).
    
    Args:
        num_points (int): Number of constellation points.
    
    Returns:
        list: Non-symmetric labels (e.g., [0, 1] for BPSK).
    """
    return list(range(NUM_OF_POINTS))
    
def labels_insertion(NUM_OF_POINTS,data, label_type="symmetric"):
    """
    Assign labels to constellation points based on their position.
    
    Args:
        num_points (int): Number of constellation points.
        data (np.ndarray): Complex-valued symbols.
        label_type (str): 'symmetric' ([1, -1]) or 'nonsymmetric' ([0, 1]).
    
    Returns:
        list: Labels for each symbol in data.
    """
    if label_type == "symmetric":
        labels = labels_creation_Symmetric(NUM_OF_POINTS)  # [1, -1]
    else:
        labels = labels_creation_NonSymmetric(NUM_OF_POINTS)  # [0, 1]
        
    assigned_labels = []
    reference_symbol = []
    for x in range(NUM_OF_POINTS):
        x_degrees = (x*360/NUM_OF_POINTS +  0) 
        x_radians = x_degrees*np.pi/180.0
        reference_symbol.append(np.cos(x_radians) + 1j*np.sin(x_radians))
    
    for d in data:
        for i in range(0,NUM_OF_POINTS):
            if np.isclose(d.real, reference_symbol[i].real, atol=1e-8) and \
            np.isclose(d.imag, reference_symbol[i].imag, atol=1e-8):
                assigned_labels.append(labels[i])
                break
        else:
            assigned_labels.append(labels[0])  # Fallback for unmatched points
    return assigned_labels
    
def data_separation(data):
    """
    Separate complex data into real and imaginary components.
    
    Args:
        data (np.ndarray): Complex-valued symbols.
    
    Returns:
        np.ndarray: 2D array with real and imaginary parts.
    """
    data = np.array(data)
    return np.stack((np.real(data), np.imag(data)), axis=1)

def create_data_set( phase_noise = 0 , noise_power = 0 , fading_scale=0):
    """
    Generate BPSK symbols with optional noise and fading.
    n and x_int can be put outside and the seed can be taken out, it was inseted here for replicability 
    Args:
        phase_noise (float): Phase noise in radians (default: 0).
        noise_power (float): Power of AWGN noise (default: 0).
        fading_scale (float): Scale of Rayleigh fading (default: 0).
    
    Returns:
        np.ndarray: Complex-valued BPSK symbols.
    """
    np.random.seed(4)
    x_int = np.random.randint(0, NUM_OF_POINTS, NUM_SYMBOLS)
    n = (np.random.randn(NUM_SYMBOLS) + 1j * np.random.randn(NUM_SYMBOLS)) / np.sqrt(2)
    x_degrees = (x_int*360/NUM_OF_POINTS +  0)#np.random.randint(0, 360)) % 360# 45, 135, 225, 315 degree
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    h = np.random.rayleigh(scale=fading_scale, size=NUM_SYMBOLS) if fading_scale > 0 else 1
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)*0
    x_symbols = h * x_symbols * np.exp(1j * phase_noise) + n * np.sqrt(noise_power)
    return x_symbols
