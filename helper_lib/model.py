import math
import torch
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
    
def swish(x):
    return x * torch.sigmoid(x)

class HW4_ENERGY(nn.Module):
    def __init__(self):
        super(HW4_ENERGY, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

class DiffusionSinusoidalEmbedding(nn.Module):
    """
    Embeds a scalar "how noisy is this image" value (noise variance, in
    [0, 1]) into a vector of sine/cosine features at different frequencies,
    the same technique as SinusoidalTimeEmbedding-style position encodings.
    """
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class DiffusionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DiffusionDownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(DiffusionResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class DiffusionUpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(DiffusionResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class HW4_DIFFUSION(nn.Module):
    """
    U-Net that predicts the noise added to a CIFAR-10 image, conditioned on
    the noise variance at the current diffusion step. image_size=32,
    num_channels=3 for CIFAR-10 (the original class-activity version
    defaulted image_size to 64 for a different dataset).
    """
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = DiffusionSinusoidalEmbedding(num_frequencies=16)

        self.down1 = DiffusionDownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DiffusionDownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DiffusionDownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = DiffusionResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = DiffusionResidualBlock(in_channels=128, out_channels=128)

        self.up1 = DiffusionUpBlock(96, in_channels=128, block_depth=2)
        self.up2 = DiffusionUpBlock(64, block_depth=2, in_channels=96)
        self.up3 = DiffusionUpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest'
        )
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


def get_model(model_name):

    if model_name.lower() == "hw2_cnn":
        model = HW2_CNN()
    elif model_name.lower() == "hw3_generator":
        model = HW3_GENERATOR()
    elif model_name.lower() == "hw3_discriminator":
        model = HW3_DISCRIMINATOR()
    elif model_name.lower() == "hw4_energy":
        model = HW4_ENERGY()
    elif model_name.lower() == "hw4_diffusion":
        model = HW4_DIFFUSION()
    else:
        raise ValueError("Only 'hw2_cnn', 'hw3_generator', 'hw3_discriminator', 'hw4_energy', and 'hw4_diffusion' models are currently implemented.")

    return model