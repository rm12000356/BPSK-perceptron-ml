"""
Configuration parameters for the 8-PSK perceptron analyzer.
These constants define the modulation, dataset size, and training settings.
"""
NUM_OF_POINTS = 8  # Number of constellation points (8 for 8-PSK)
NUM_SYMBOLS = 100  # Number of symbols in the dataset
NUM_EPOCHS = 10000  # Maximum training epochs
LEARNING_RATE = 0.05  # Learning rate for perceptron
NOISE_POWER = 0.4  # Power of AWGN noise
PHASE_NOISE = 0.2  # Phase noise in radians
FADING_SCALE = 0.5  # Scale of Rayleigh fading
a = 1  # Used in tanh activation (a * tanh(b * x))
b = 1  # Used in tanh activation
INPUT_DIM = 2  # Input dimension (real and imaginary parts)
NUM_HIDDEN_LAYERS = 2
HIDDEN_DIMS = [round(NUM_OF_POINTS * 1.5), round(NUM_OF_POINTS * 1.2)]  # Neurons per hidden layer, e.g., [12, 10]


