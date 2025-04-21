import torch
import torch.nn as nn

# Define ChurnModel here directly
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        return x

# Register the class with PyTorch's pickle loader
torch.serialization.add_safe_globals([ChurnModel])

# Load full model
full_model = torch.load(
    "/media/jai/Projects/projects/ai-churn/CIS579_Churn/bharath/temp_model.pth",
    map_location="cpu",
    weights_only=False
)

# Save only weights
torch.save(full_model.state_dict(), "/media/jai/Projects/projects/ai-churn/CIS579_Churn/bharath/ann_weights_only.pth")
print("Weights-only model saved successfully.")
