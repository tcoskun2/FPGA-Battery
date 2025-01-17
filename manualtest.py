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
model = (torch.load("testingmodel.pt"))

# Define input
test_input = [2.6243, 998.8, 1032.187, 2553.117]

# Convert to NumPy array
test_input_array = np.array(test_input, dtype=np.float32)

# Extract weights and biases
weights_and_biases = []
for name, param in model.named_parameters():
    if param.requires_grad:
        weights_and_biases.append(param.detach().cpu().numpy())

# Unpack the weights and biases
W1, b1 = weights_and_biases[0], weights_and_biases[1]
W2, b2 = weights_and_biases[2], weights_and_biases[3]
W3, b3 = weights_and_biases[4], weights_and_biases[5]
W4, b4 = weights_and_biases[6], weights_and_biases[7]

def relu(x):
    return np.maximum(0, x)

# Layer 1
z1 = np.dot(test_input_array, W1.T) + b1
a1 = relu(z1)

#print(W1.T)
print(a1)

# Layer 2
z2 = np.dot(a1, W2.T) + b2
a2 = relu(z2)

print(a2)

# Layer 3
z3 = np.dot(a2, W3.T) + b3
a3 = relu(z3)

print(a3)
print(weights_and_biases[6])

# Layer 4 (output layer)
z4 = np.dot(a3, W4.T) + b4
output = z4  # No activation for the output layer

# Print result
print(f"Manual prediction for input {test_input}: {output[0]:.4f}")

# Define input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)


test_input_tensor = test_input_tensor.unsqueeze(0)  # Shape becomes [1, 4]

model.eval()

with torch.no_grad():
    prediction = model(test_input_tensor)

    # Output the prediction
print(f"Prediction for input {test_input}: {prediction.item():.4f}")


