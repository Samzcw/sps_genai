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
    
class HW3_GENERATOR(nn.Module):
    def __init__(self, z_dim=100):
        super(HW3_GENERATOR, self).__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.reshape = lambda x: x.view(-1, 128, 7, 7)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x
    
class HW3_DISCRIMINATOR(nn.Module):
    def __init__(self):
        super(HW3_DISCRIMINATOR, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

def get_model(model_name):

    if model_name.lower() == "hw2_cnn":
        model = HW2_CNN()
    elif model_name.lower() == "hw3_generator":
        model = HW3_GENERATOR()
    elif model_name.lower() == "hw3_discriminator":
        model = HW3_DISCRIMINATOR()
    else:
        raise ValueError("Only 'hw2_cnn', 'hw3_generator', and 'hw3_discriminator' models are currently implemented.")

    return model