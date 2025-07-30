"""
Utility functions for plotting 8-PSK constellation diagrams.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_utils import data_separation

def get_label_color_pairs(labels):
    """Generate unique color for each label."""
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
    return list(zip(unique_labels, colors))

def plot_graph(x, y, weights, binary_target, title="8-PSK Constellation"):
    """Plot 8-PSK constellation with true labels."""
    # Convert complex x to real/imag if needed
    x = data_separation(x, 0) if x.dtype == np.complex128 else x
    # Get colors for symmetric labels (e.g., [1, -1, 3, -3, 5, -5, 7, -7])
    color_pairs = get_label_color_pairs(y)
    color_dict = dict(color_pairs)
    true_colors = [color_dict[label] for label in y]
    
    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], c=true_colors, s=50, marker='o', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel("In-phase (I)", fontsize=12)
    plt.ylabel("Quadrature (Q)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()