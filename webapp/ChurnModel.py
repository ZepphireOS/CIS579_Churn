import torch
import torch.nn as nn

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_dim, 128)  # Input layer to first hidden layer
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)         # First hidden layer to second hidden layer
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(128, 1)          # Second hidden layer to output layer

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        # Pass input through the network
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))  # Apply ReLU after first layer
        #x = self.dropout(self.relu(self.bn2(self.fc2(x))))  # Apply ReLU after second layer
        x = self.fc3(x)  # Output using Sigmoid activation
        return x