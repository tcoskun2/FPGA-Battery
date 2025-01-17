import torch
import numpy as np
import torch.nn as nn

class MultiInputRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(MultiInputRegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Load the trained weights
model = MultiInputRegressionModel(input_size=4)
model = (torch.load("trained_weights.pth",weights_only=True))

# Define a function to convert a number to fixed-point representation
def to_fixed_point(value, total_bits=32, frac_bits=16):
    scale = 2 ** frac_bits
    fixed_point = int(value * scale) & ((1 << total_bits) - 1)
    return fixed_point

# Extract and save weights and biases
for i, layer in enumerate(model.network):
    if isinstance(layer, torch.nn.Linear):
        weights = layer.weight.detach().numpy()
        biases = layer.bias.detach().numpy()

        # Save weights
        np.savetxt(f"layer{i}_weights.txt", weights, fmt='%d', delimiter=",",
                   header=f"Fixed-point weights for layer {i} (rows are output neurons)")

        # Save biases
        np.savetxt(f"layer{i}_biases.txt", biases, fmt='%d', delimiter=",",
                   header=f"Fixed-point biases for layer {i}")
