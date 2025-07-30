# 8-PSK Multilayer Perceptron Classifier
## Overview
This project implements a multilayer perceptron (MLP) to classify 8-PSK (Phase Shift Keying) signals under various conditions, including noise, phase noise, and Rayleigh fading. The MLP is trained to map complex-valued 8-PSK symbols to symmetric labels (e.g., [1, -1, 3, -3, 5, -5, 7, -7]) encoded in binary form. The project includes data generation, training, testing, and visualization of constellation diagrams with decision boundaries.
## Project Structure

data_utils.py: Generates 8-PSK datasets, assigns symmetric or non-symmetric labels, and separates complex data into real/imaginary components.
config.py: Defines configuration parameters (e.g., NUM_OF_POINTS = 8, NUM_SYMBOLS = 100, LEARNING_RATE = 0.05, OUTPUT_DIM = 4).
MultiLayerPerceptron_model.py: Implements the MLP with two hidden layers, training with backpropagation, and testing with MSE and accuracy metrics.
plot_utils.py: Plots 8-PSK constellations with true labels and adaptive decision boundaries using the trained MLP weights.
main.py: Orchestrates data generation, training, testing, and plotting for five conditions: pure, noise, phase, both, and fading.
run_main.ipynb: Jupyter notebook to run main.py and display results.

## Features

Modulation: 8-PSK with symmetric labels and binary encoding.
MLP Architecture: Two hidden layers with 12 and 10 neurons, tanh and sigmoid activations.
Conditions Tested: Pure signal, AWGN noise, phase noise, combined noise/phase, and Rayleigh fading.
Visualization: Constellation plots with decision boundaries for each condition.
Metrics: Mean squared error (MSE) and classification accuracy.

## Requirements

Python 3.10
Libraries: numpy, matplotlib
Environment: Tested in a dl_env virtual environment (see run_main.ipynb).

## Setup

Install dependencies:pip install numpy matplotlib


Ensure Python 3.10 is installed and activate your environment:conda activate dl_env


Place all .py files and run_main.ipynb in the same directory.

## Usage

Open run_main.ipynb in Jupyter Notebook.
Run the cell containing %run main.py.
Expected output:
Training stops when MSE < 0.01 (e.g., Stopping early at epoch X with MSE Y.YYYY).
For each condition: [name] error = X.XXXX, accuracy = Y.YYYY.
Five plots showing 8-PSK constellations with symmetric labels and decision boundaries.



## Notes

Labels: Symmetric labels ([1, -1, 3, -3, 5, -5, 7, -7]) are used with 4-bit binary encoding.
Data: 100 symbols per dataset (NUM_SYMBOLS = 100).
Plots: Decision boundaries are adaptive, based on MLP predictions over a grid.
Tuning: Adjust LEARNING_RATE, NUM_EPOCHS, or mse_threshold in config.py for better performance.

## Future Improvements

Add non-symmetric label support.
Enhance decision boundary resolution.
Increase NUM_SYMBOLS for more robust training.
Add docstrings for all functions in data_utils.py.
