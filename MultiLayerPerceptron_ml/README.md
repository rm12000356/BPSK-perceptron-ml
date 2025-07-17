# BPSK multilayer Perceptron ML Model

A lightweight and adaptive signal classifier that uses a multilayer-layer perceptron to learn BPSK transitions from raw I/Q signal data.
What makes this project unique is not just the model — it’s the way data is generated, paired, and labeled to simulate real signal dynamics. There’s also the ability to add noise and phase shifts to simulate reception in different channels with side information (CSI).



## Overview

This is a simple multilayer-layer perceptron. It’s implemented like any basic perceptron: you give it data (labeled data), and if the data is linearly separable, it can learn to separate it.

There are options for plotting, data creation (with different parameters), and more. In this project, the model works only for 64 points in a constellation, if noice,phase shift are introduce the optimal is 8, and still errors are detected.

The result:
You get a basic model that learns to identify non-linearly separable points — the main point is that even if i got the right code and i have nearly everythinv vorect the enviroment affects everything, i need to have away to let the machine nor identify the channel side information.

## Key Features

- multilayer-layer perceptron — implemented fully from scratch (no ML libraries)
- Real-time visualization — plots learning progress and predictions
- Modular codebase — clean separation of model, data, and plotting

## How Data Works

A sample implementation is provided in **main**. Feedback and suggestions for enhancements to this project are welcome and greatly appreciated.



