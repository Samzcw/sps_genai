import torch.nn as nn
import torch.nn.functional as F

class HW2_CNN(nn.Module):
    def __init__(self):
        super(HW2_CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )  # Input channels = 3, Output channels = 16
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )  # Input channels = 16, Output channels = 32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer
        self.fc2 = nn.Linear(100, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(model_name):

    if model_name.lower() == "hw2_cnn":
        model = HW2_CNN()
    else:
        raise ValueError("Only 'hw2_cnn' model is currently implemented.")

    return model