## BPSK Perceptron Analyzer

This project implements a perceptron-based analyzer for Binary Phase Shift Keying (BPSK) modulation. It generates BPSK symbols, applies noise and fading effects, trains a perceptron to classify constellation points, and visualizes the results with decision boundaries and error analysis.

## Features

Generates BPSK symbols with configurable noise (AWGN, phase noise) and Rayleigh fading.
Trains a perceptron with symmetric ([1, -1]) or non-symmetric ([0, 1]) labels.
Visualizes constellation diagrams, decision boundaries, and Rayleigh fading histograms.
Evaluates classification performance under various conditions (pure, noise, phase, fading).
Configurable parameters for dataset size, epochs, learning rate, and noise levels.

## Usage

1.Run the main script:Execute the main script to generate data, train the perceptron, and visualize results:
```
python main.py
```
This will output error rates for different scenarios (pure, noise, phase, both, fading) and display plots.

2.Run the Jupyter notebook:Alternatively, use the provided notebook for interactive execution:
```
jupyter notebook run_main.ipynb
```
Run all cells to see the same outputs as main.py.

3.Customize parameters:Modify config.py to adjust:

NUM_SYMBOLS: Number of symbols (default: 1000).
NUM_EPOCHS: Training epochs (default: 1000).
LEARNING_RATE: Perceptron learning rate (default: 0.01 for stability).
NOISE_POWER, PHASE_NOISE, FADING_SCALE: Noise and fading parameters.

## File Structure

config.py: Configuration parameters (e.g., number of symbols, epochs, noise levels).
main.py: Main script to run the BPSK analyzer, including data generation, training, and visualization.
data_utils.py: Functions for generating BPSK symbols and labels.
perceptron_model.py: Perceptron training and testing logic.
plot_utils.py: Visualization functions for constellations and histograms.
run_main.ipynb: Jupyter notebook for interactive execution and visualization.

## Example Output
Running main.py or run_main.ipynb produces:

Console output: Error counts and rates for each scenario (e.g., Pure: errors=0, error_rate=0.00%).
Plots:
Rayleigh fading histogram.
Constellation diagrams with decision boundaries for each scenario.
Errors highlighted in red for misclassified points.



## Notes

The perceptron is sensitive to initial weights. The current implementation uses zero-initialized weights ([0, 0]) for stability, ensuring a vertical decision boundary (x=0) for clean BPSK data.
Numerical noise in imaginary components is minimized by forcing x_symbols = np.cos(x_radians) + 0j in data_utils.py.
For noisy data (e.g., AWGN, phase noise, fading), error rates increase, and the decision boundary may shift slightly.

## Requirements 

Python 3.10+
NumPy
Matplotlib
Jupyter Notebook (optional, for run_main.ipynb)


Built with Python, NumPy, and Matplotlib.
Inspired by digital communication systems and machine learning concepts.
