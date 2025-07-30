"""
Main script for training and testing a multilayer perceptron on 8-PSK signals.
Generates datasets with noise, phase noise, and fading, then evaluates the MLP.
"""
from data_utils import create_data_set, labels_insertion
from config import NUM_OF_POINTS, NOISE_POWER, PHASE_NOISE, FADING_SCALE
from MultiLayerPerceptron_model import training_multilayer, testing
from plot_utils import plot_graph

# Define test conditions
conditions = [
    {"name": "pure", "noise_power": 0, "phase_noise": 0, "fading_scale": 0},
    {"name": "noise", "noise_power": NOISE_POWER, "phase_noise": 0, "fading_scale": 0},
    {"name": "phase", "noise_power": 0, "phase_noise": PHASE_NOISE, "fading_scale": 0},
    {"name": "both", "noise_power": NOISE_POWER / 2, "phase_noise": PHASE_NOISE, "fading_scale": 0},
    {"name": "fading", "noise_power": 0, "phase_noise": 0, "fading_scale": FADING_SCALE}
]

# Generate training data and labels
Data = create_data_set()
target = labels_insertion(NUM_OF_POINTS, Data)

# Train the multilayer perceptron
weights, binary_target = training_multilayer(Data, target, mse_threshold=0.01)

# Test and plot for each condition
for cond in conditions:
    # Filter out 'name' key for create_data_set
    data_params = {k: v for k, v in cond.items() if k in ['noise_power', 'phase_noise', 'fading_scale']}
    test_data = create_data_set(**data_params)
    mse, accuracy = testing(test_data, weights, binary_target, rounding=1)
    print(f"{cond['name']} error = {mse:.4f}, accuracy = {accuracy:.4f}")
    plot_graph(test_data, target, weights, binary_target, title=f"8-PSK Constellation ({cond['name']})")