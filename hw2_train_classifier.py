from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.checkpoints import load_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hw2_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load data
train_loader = get_data_loader(
    "data/hw2",
    batch_size=32,
    split="train",
    dataset_name="CIFAR10",
    transform=hw2_transform,
    val_ratio=0.1,
    seed=42
)

val_loader = get_data_loader(
    "data/hw2",
    batch_size=32,
    split="val",
    dataset_name="CIFAR10",
    transform=hw2_transform,
    val_ratio=0.1,
    seed=42
)

test_loader = get_data_loader(
    "data/hw2",
    batch_size=32,
    split="test",
    dataset_name="CIFAR10",
    transform=hw2_transform,
    seed=42
)

# Initialize model and training components
hw2_model = get_model("hw2_cnn").to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(hw2_model.parameters(), lr=0.001)

hw2_trained_model, hw2_best_path, hw2_datalogs = train_model(
    model=hw2_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10,
    checkpoint_dir="hw2_checkpoints"
)

print(f"Best checkpoint path: {hw2_best_path}")

hw2_best_model = hw2_trained_model

load_checkpoint(
    hw2_best_model,
    optimizer,
    hw2_best_path,
    device=device
)

# Evaluate best model
avg_loss, accuracy = evaluate_model(
    hw2_best_model,
    test_loader,
    criterion,
    device=device
)

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")