import numpy as np
from data_utils import create_data_set, labels_insertion
from perceptron_model import train_perceptron, test_training
from plot_utils import plot_graph, plot_error_graph, plot_histogram
from config import NUM_OF_POINTS, NOISE_POWER, PHASE_NOISE, FADING_SCALE, NUM_SYMBOLS

def main():
    """
    Run BPSK perceptron analyzer with training, testing, and visualization.
    """
    np.random.seed(4)
    label_type = "symmetric"
    
    # Generate data
    Data = create_data_set()
    Data_with_noise = create_data_set(noise_power=NOISE_POWER)
    Data_with_phase = create_data_set(phase_noise=PHASE_NOISE)
    Data_with_both = create_data_set(phase_noise=PHASE_NOISE, noise_power=NOISE_POWER)
    Data_with_fading = create_data_set(fading_scale=FADING_SCALE)
    target = labels_insertion(NUM_OF_POINTS, Data, label_type)
    
    # Plot histogram
    plot_histogram(FADING_SCALE)
    
    # Train perceptron
    weights = train_perceptron(Data, label_type)
    # Test and plot
    scenarios = [
        ("Pure", Data, target),
        ("Noise", Data_with_noise, target),
        ("Phase", Data_with_phase, target),
        ("Both", Data_with_both, target),
        ("Fading", Data_with_fading, target)
    ]
    
    for name, data, tgt in scenarios:
        error, errors = test_training(data, weights, tgt, label_type)
        error_rate = error / NUM_SYMBOLS * 100
        print(f"{name}: errors={error}, error_rate={error_rate:.2f}%")
        plot_error_graph(data, tgt, errors, weights)
        # Optionally save plots
        # plt.savefig(f"{name.lower()}_constellation.png")
    
    # Optional: Test non-symmetric labels
    label_type_non_sym="nonsymmetric"
    target_non_sym = labels_insertion(NUM_OF_POINTS, Data, "nonsymmetric")
    
    weights_non_sym = train_perceptron(Data, "nonsymmetric")
    scenarios_non_sym = [
        ("Pure", Data, target_non_sym),
        ("Noise", Data_with_noise, target_non_sym),
        ("Phase", Data_with_phase, target_non_sym),
        ("Both", Data_with_both, target_non_sym),
        ("Fading", Data_with_fading, target_non_sym)
    ]
    for name, data, tgt in scenarios_non_sym:
        error, errors = test_training(data, weights_non_sym, tgt, label_type_non_sym)
        error_rate = error / NUM_SYMBOLS * 100
        print(f"{name}: errors={error}, error_rate={error_rate:.2f}%")
        plot_error_graph(data, tgt, errors, weights_non_sym)
    
if __name__ == "__main__":
    main()