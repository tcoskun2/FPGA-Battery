import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import time
import numpy as np


import hls4ml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Running on {device}')

# Load the dataset
data_file = "C:/Users/8BitO/Desktop/FPGALLM/FinalData_2500V/FinalData_2500V.xlsx"
df = pd.read_excel(data_file, engine='openpyxl')
df = df[['Voltage(V)', 'Current(mA)', 'Capacity(mAh)', 'Energy(mWh)', 'SoC']]

# Split data into features and target
X = df[['Voltage(V)', 'Current(mA)', 'Capacity(mAh)', 'Energy(mWh)']].values
y = df['SoC'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the dataset class
class BatteryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }

# Create data loaders
BATCH_SIZE = 200
train_data_loader = DataLoader(BatteryDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_data_loader = DataLoader(BatteryDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Define the model
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

# Initialize the model
model = MultiInputRegressionModel(input_size=4).to(device)

# Training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Track training time
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            predictions = model(features)
            loss = loss_fn(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                predictions = model(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        mse = mean_squared_error(all_labels, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_labels, all_predictions)
        mae = mean_absolute_error(all_labels, all_predictions)
        mean_true_value = np.mean(all_labels)
        accuracy = (1 - mae / mean_true_value) * 100

        logger.info(f"Epoch {epoch + 1}, Test MSE: {mse:.4f}")
        logger.info(f"Epoch {epoch + 1}, Test RMSE: {rmse:.4f}")
        logger.info(f"Epoch {epoch + 1}, Test R² Score: {r2:.4f}")
        logger.info(f"Epoch {epoch + 1}, Test MAE: {mae:.4f}")
        logger.info(f"Epoch {epoch + 1}, Approximate Accuracy: {accuracy:.2f}%")

    # Total training time
    total_training_time = time.time() - start_time
    logger.info(f"Total Training Time: {total_training_time:.2f} seconds")

    # Calculate model complexity (number of parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {total_params}")

    # Inference time
    start_inference_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            _ = model(features)
    total_inference_time = time.time() - start_inference_time
    logger.info(f"Total Inference Time: {total_inference_time:.2f} seconds")


def float_to_fixed_32bit(value):
    # number of fractional bits
    fractional_bits = 16

    # Scale
    scaled_value = round(value * (1 << fractional_bits))

    # Ensure value fits within 32-bit signed integer range
    if scaled_value < -2147483648:
        scaled_value = -2147483648
    elif scaled_value > 2147483647:
        scaled_value = 2147483647

    # Convert to 32-bit signed integer
    fixed_point_value = int(scaled_value) & 0xFFFFFFFF

    return fixed_point_value

def to_binary_32bit(value):
    # Convert the 32-bit value to binary and pad to 32 bits
    return f"{value:032b}"

def export_weights_to_files(model, output_dir="weights_biases_files200"):
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, params in model.state_dict().items():
        param_array = params.cpu().detach().numpy()

        if len(param_array.shape) == 2:  # For weights (2D array)
            file_path = os.path.join(output_dir, f"{layer_name}_weights.txt")
            with open(file_path, "w") as f:
                for row in param_array:
                    fixed_values = "".join(to_binary_32bit(float_to_fixed_32bit(val)) for val in row)
                    f.write(f"{fixed_values}\n")

        elif len(param_array.shape) == 1:  # For biases (1D array)
            file_path = os.path.join(output_dir, f"{layer_name}_biases.txt")
            with open(file_path, "w") as f:
                fixed_values = "".join(to_binary_32bit(float_to_fixed_32bit(val)) for val in param_array)
                f.write(f"{fixed_values}\n")



if __name__ == '__main__':
    train_and_evaluate(model, train_data_loader, test_data_loader, epochs=12)

    # testing
    test_input = [2.4767, 999, 599.3133, 1449]


    test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
    test_input_tensor = test_input_tensor.unsqueeze(0)  
    model.eval()
    with torch.no_grad():
        prediction = model(test_input_tensor)

    # Output 
    logger.info(f"Prediction for input {test_input}: {prediction.item():.4f}")
    print(f"Prediction for input {test_input}: {prediction.item():.4f}")

    export_weights_to_files(model)
    
    torch.save(model.state_dict(), "testingweight.pth")

    torch.save(model, "testingmodel.pt")
    
