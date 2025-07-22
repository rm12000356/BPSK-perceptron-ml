import matplotlib.pyplot as plt
import numpy as np
from config import NUM_SYMBOLS

def get_label_color_pairs(labels):
    """
    Generate color pairs for unique labels.
    
    Args:
        labels (list): List of labels.
    
    Returns:
        list: List of (label, color) tuples, where color is an RGB array.
    """
    np.random.seed(4)
    unique_labels = sorted(set(labels))
    label_color_pairs = []

    for label in unique_labels:
        color = np.random.rand(3,) *0.8  # Random RGB color
        label_color_pairs.append((label, color))

    return label_color_pairs

def plot_graph(x, y):
    """
    Plot BPSK constellation without decision boundary.
    
    Args:
        x (np.ndarray): Complex-valued symbols.
        y (list): Labels for each symbol.
    """
    color = get_label_color_pairs(y)
    color_dict = dict(color)
    colors = [color_dict[label] for label in y]
    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(x), np.imag(x), c=colors, alpha=0.5)
    plt.title("BPSK Constellation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')  
    plt.show()

def plot_error_graph(x,y,errors, weights):
    """
    Plot BPSK constellation with decision boundary and error highlights.
    
    Args:
        x (np.ndarray): Complex-valued symbols.
        y (list): Labels for each symbol.
        errors (np.ndarray): Binary array indicating misclassifications (1=error).
        weights (np.ndarray): Perceptron weights for decision boundary.
    """
    color_dict = dict(get_label_color_pairs(y))
    colors = ['red' if errors[i] == 1 else color_dict[y[i]] for i in range(len(y))]
    plt.figure(figsize=(5, 5))
    
    I_min, I_max = np.min(np.real(x)), np.max(np.real(x))
    I_range = np.linspace(I_min, I_max, 200)
    
    if abs(weights[1]) > 1e-6:
        Q_vals = -(weights[0] * I_range) / weights[1]
        Q_min, Q_max = np.min(np.imag(x)), np.max(np.imag(x))
        Q_vals = np.clip(Q_vals, Q_min - 1, Q_max + 1)
        plt.plot(I_range, Q_vals, 'k--', label='Decision Boundary')
    else:
        x_intercept = 0 if abs(weights[0]) < 1e-6 else -weights[1] / weights[0]
        plt.axvline(x=x_intercept, color='k', linestyle='--', label='Decision Boundary')
    
    plt.scatter(np.real(x), np.imag(x), c=colors, alpha=0.5)
    plt.title("BPSK Constellation with Decision Boundary")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()    

     

def plot_histogram(fading_scale=0.5):
    """
    Plot histogram of Rayleigh fading amplitudes.
    
    Args:
        fading_scale (float): Scale of Rayleigh fading (default: 0.5).
    """
    np.random.seed(4)
    h = np.random.rayleigh(scale=fading_scale, size = NUM_SYMBOLS)
    plt.hist(h, bins=30, density=True)
    plt.title(f"Rayleigh Fading Histogram (scale={fading_scale})")
    plt.xlabel("Fading Amplitude")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()