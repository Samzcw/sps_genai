from helper_lib.data_loader import get_data_loader
from helper_lib.gan_trainer import train_gan
from helper_lib.model import get_model

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalize to [-1, 1] to match the generator's tanh output
hw3_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# GANs don't need a train/val/test split -- use every available image
dataloader = get_data_loader(
    "data/hw3",
    batch_size=128,
    split=None,
    dataset_name="MNIST",
    transform=hw3_transform
)

hw3_gen = get_model("hw3_generator").to(device)
hw3_disc = get_model("hw3_discriminator").to(device)

hw3_trained_gen, hw3_trained_disc, hw3_datalogs = train_gan(
    generator=hw3_gen,
    discriminator=hw3_disc,
    dataloader=dataloader,
    device=device,
    z_dim=100,
    lr=2e-4,
    epochs=10,
    checkpoint_dir='checkpoints',
    sample_dir='samples',
    log_every=100
)

print("Training complete. Saving final models...")
torch.save(hw3_trained_gen.state_dict(), "checkpoints/hw3_generator_final.pth")
