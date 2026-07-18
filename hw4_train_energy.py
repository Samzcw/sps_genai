from helper_lib.data_loader import get_data_loader
from helper_lib.energy_trainer import train_energy_model
from helper_lib.model import get_model

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalize to [-1, 1] -- Langevin dynamics sampling clamps images to this range
hw4_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = get_data_loader(
    "data/hw4",
    batch_size=128,
    split="train",
    dataset_name="CIFAR10",
    transform=hw4_transform,
    val_ratio=0.1,
    seed=42
)

val_loader = get_data_loader(
    "data/hw4",
    batch_size=128,
    split="val",
    dataset_name="CIFAR10",
    transform=hw4_transform,
    val_ratio=0.1,
    seed=42
)

hw4_energy_model = get_model("hw4_energy").to(device)

hw4_trained_energy, hw4_datalogs = train_energy_model(
    model=hw4_energy_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=10,
    lr=1e-4,
    alpha=0.1,
    steps=60,
    step_size=10,
    noise=0.005,
    img_shape=(3, 32, 32),
    checkpoint_dir='hw4_checkpoints',
    sample_dir='hw4_samples'
)

print("Training complete. Saving final model...")
torch.save(hw4_trained_energy.state_dict(), "hw4_checkpoints/hw4_energy_final.pth")