# BPSK Perceptron ML Model

A lightweight and adaptive signal classifier that uses a single-layer perceptron to learn BPSK transitions from raw I/Q signal data.
What makes this project unique is not just the model — it’s the way data is generated, paired, and labeled to simulate real signal dynamics. There’s also the ability to add noise and phase shifts to simulate reception in different channels with side information (CSI).



## Overview

This is a simple single-layer perceptron. It’s implemented like any basic perceptron: you give it data (labeled data), and if the data is linearly separable, it can learn to separate it.

There are options for plotting, data creation (with different parameters), and more. In this project, the model works only for two points in a constellation. The next model I’ll be working on is a multilayer perceptron that can detect more points in a constellation.

The result:
You get a basic model that learns to identify linearly separable points — but the real value here is in the data creation and manipulation, which is the main focus of this project and will be reused and improved in future versions.

## Key Features

- Single-layer perceptron — implemented fully from scratch (no ML libraries)
- Signal-aware data preparation — uses distance and transitions between I/Q points
- Real-time visualization — plots learning progress and predictions
- Modular codebase — clean separation of model, data, and plotting

## How Data Works

A sample implementation is provided in **main**. Feedback and suggestions for enhancements to this project are welcome and greatly appreciated.



