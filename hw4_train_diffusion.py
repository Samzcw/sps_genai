from helper_lib.data_loader import get_data_loader
from helper_lib.diffusion_trainer import DiffusionModel, offset_cosine_diffusion_schedule, train_diffusion
from helper_lib.model import get_model

import torch
import torch.nn as nn
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 64

# Diffusion normalizes internally using real per-channel stats (computed
# below), so we just need raw [0, 1] pixel values here -- no [-1, 1] mapping
# like the GAN/energy models used.
hw4_diffusion_transform = transforms.ToTensor()

train_loader = get_data_loader(
    "data/hw4",
    batch_size=BATCH_SIZE,
    split="train",
    dataset_name="CIFAR10",
    transform=hw4_diffusion_transform,
    val_ratio=0.1,
    seed=42
)

val_loader = get_data_loader(
    "data/hw4",
    batch_size=BATCH_SIZE,
    split="val",
    dataset_name="CIFAR10",
    transform=hw4_diffusion_transform,
    val_ratio=0.1,
    seed=42
)

# Compute real per-channel normalization statistics from the training data
print("Computing normalization statistics...")
mean = torch.zeros(NUM_CHANNELS)
std = torch.zeros(NUM_CHANNELS)
total_samples = 0

for imgs, _ in train_loader:
    batch_size = imgs.size(0)
    imgs_flat = imgs.view(batch_size, NUM_CHANNELS, -1)
    batch_mean = imgs_flat.mean(dim=(0, 2))
    batch_std = imgs_flat.std(dim=(0, 2))
    mean += batch_mean * batch_size
    std += batch_std * batch_size
    total_samples += batch_size

mean /= total_samples
std /= total_samples
print("Normalization stats - Mean:", mean, "Std:", std)
mean = mean.reshape(1, NUM_CHANNELS, 1, 1).to(device)
std = std.reshape(1, NUM_CHANNELS, 1, 1).to(device)

hw4_unet = get_model("hw4_diffusion").to(device)
hw4_diffusion_model = DiffusionModel(hw4_unet, offset_cosine_diffusion_schedule, ema_decay=0.999)
hw4_diffusion_model.set_normalizer(mean, std)

optimizer = torch.optim.AdamW(hw4_diffusion_model.network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.L1Loss()

train_diffusion(
    hw4_diffusion_model, train_loader, val_loader, optimizer, loss_fn,
    epochs=10, device=device, image_size=IMAGE_SIZE, diffusion_steps=20,
    checkpoint_dir='hw4_checkpoints', sample_dir='hw4_samples_diffusion'
)

print("Training complete. Saving final EMA model...")
torch.save({
    "ema_model_state_dict": hw4_diffusion_model.ema_network.state_dict(),
    "normalizer_mean": hw4_diffusion_model.normalizer_mean,
    "normalizer_std": hw4_diffusion_model.normalizer_std,
}, "hw4_checkpoints/hw4_diffusion_final.pth")
