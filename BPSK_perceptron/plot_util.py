import matplotlib.pyplot as plt
import numpy as np

def get_label_color_pairs(labels):
    unique_labels = sorted(set(labels))
    label_color_pairs = []

    for label in unique_labels:
        color = np.random.rand(3,) *0.8  # Random RGB color
        label_color_pairs.append((label, color))

    return label_color_pairs

def plot_grapgh(x, y):
    color = get_label_color_pairs(y)
    color_dict = dict(color)
    colors = [color_dict[label] for label in y]
    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(x), np.imag(x), c=colors, alpha=0.5)
    plt.title("BPSK Constellation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio
    plt.show()