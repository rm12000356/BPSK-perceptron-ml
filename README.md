# 🧠 BPSK Perceptron ML Model

A lightweight and adaptive signal classifier that uses a **single-layer perceptron** to learn BPSK transitions from raw I/Q signal data.  
What makes this project unique is not just the model — it’s the way **data is generated, paired, and labeled** to simulate real signal dynamics.

---

## 📌 Overview

Instead of labeling I/Q points statically, this model **compares consecutive signal samples** to detect transitions — mimicking how real receivers operate.

The result:  
A minimal but intelligent classifier that can understand binary phase shifts in a BPSK signal by **learning from the way the signal moves**, not just where it is.

---

## 🎯 Key Features

- 🧠 **Single-layer perceptron** — implemented fully from scratch (no ML libraries)
- 📡 **Signal-aware data preparation** — uses distance and transitions between I/Q points
- 🔁 **Dynamic thresholding** — classifies transitions based on Euclidean movement
- 📊 **Real-time visualization** — plots learning progress and predictions
- 🛠️ **Modular codebase** — clean separation of model, data, and plotting

---

## 🧬 How Data Works

This is the core innovation of the project:

```python
# Example signal points (I/Q):
[ (0.9, 0.1), (0.95, 0.05), (-0.9, -0.1), (-0.95, -0.05) ]

# Step 1: Pair consecutive points
# → [(pt1, pt2), (pt2, pt3), ...]

# Step 2: Measure distance
# If distance > threshold → label = 1 (symbol flip)
# Else                   → label = 0 (same symbol)

# Step 3: Feed pairs and labels to perceptron

