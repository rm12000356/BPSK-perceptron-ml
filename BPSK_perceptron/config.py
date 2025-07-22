"""
Configuration parameters for the BPSK perceptron analyzer.
These constants define the modulation, dataset size, and training settings.
"""
NUM_OF_POINTS = 2  # Number of constellation points (2 for BPSK)
NUM_SYMBOLS = 1000  # Number of symbols in the dataset
NUM_EPOCHS = 1000  # Maximum training epochs 
LEARNING_RATE = 0.001 # Learning rate for perceptron 1
NOISE_POWER = 0.4  # Power of AWGN noise
PHASE_NOISE = 0.2  # Phase noise in radians
FADING_SCALE = 0.5  # Scale of Rayleigh fading
