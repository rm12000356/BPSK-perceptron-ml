# Signal Processing and Machine Learning Projects

This repository contains a collection of personal and academic projects combining signal processing techniques with machine learning models. Each subfolder focuses on a specific task or algorithm, primarily built using Python and NumPy from scratch.

## Repository Structure

### 1. BPSK-perceptron-ml/
Implements a custom singlelayer perceptron to classify BPSK (Binary Phase Shift Keying) signals based on I/Q input data. The model is adaptive, supports variable hidden layers, and includes visualization of the training process.

- Tools used: Python, NumPy, Matplotlib
- Focus: Modulation classification, neural networks, digital signal processing

### 2. MultiLayerPerceptron_ml/
Implements a custom Mutilayer perceptron to classify BPSK (Binary Phase Shift Keying)(more than 2 constelation points) signals based on I/Q input data. The model is adaptive, supports variable hidden layers, and includes visualization of the training process.

- Tools used: Python, NumPy, Matplotlib
- Focus: Modulation classification, neural networks, digital signal processing
## Goals

- Explore how machine learning models can classify modulated signals
- Build neural networks from scratch without external libraries
- Apply signal processing knowledge in practical code
- Prepare for future integration with real-world SDR data

## About

This repository is part of an ongoing study combining communication engineering, digital signal processing, and machine learning. Projects are implemented for learning purposes, academic research, and skill development.

For questions or collaboration, feel free to connect with me on LinkedIn or GitHub.

## future ideas

I plan to start working on QAM (Quadrature Amplitude Modulation) by reusing the code developed for BPSK. The goal is to adapt the existing architecture to handle QAM classification.

The next phase will involve testing various pre-built models to identify the best-performing one for this task. Once the optimal model is found, I will reimplement it from scratch to better understand its mechanics and explore possible improvements.

One area of enhancement I want to explore is designing the model to depend on multiple input features. This could enable the system to distinguish between different QAM schemes—such as identifying whether a user is using 4-QAM, 8-QAM, etc.—and make a classification accordingly. This is a more advanced goal, but one that I believe is achievable with the right approach.
