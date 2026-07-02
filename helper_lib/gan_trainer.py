import os
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from .checkpoints import save_checkpoint


def train_gan(generator, discriminator, dataloader, device='cpu', z_dim=100, lr=2e-4, epochs=1,
              checkpoint_dir='checkpoints', sample_dir='samples', log_every=100):
    """
    Train a vanilla GAN using binary cross-entropy loss.
    Mirrors the structure of trainer.train_model.
    """

    datalogs = []
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss()

    # betas=(0.5, 0.999) is the standard choice for GAN training with Adam
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Fixed noise so sample images are comparable across epochs
    fixed_noise = torch.randn(64, z_dim).to(device)

    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )

        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            ## === Train Discriminator === ##
            # Real images should be classified as real (label 1)
            disc_real = discriminator(real)
            loss_disc_real = criterion(disc_real, real_labels)

            # Fake images should be classified as fake (label 0)
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = generator(noise)
            disc_fake = discriminator(fake.detach())
            loss_disc_fake = criterion(disc_fake, fake_labels)

            loss_disc = loss_disc_real + loss_disc_fake

            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ## === Train Generator === ##
            # Generator wants the discriminator to classify fakes as real
            disc_fake_for_gen = discriminator(fake)
            loss_gen = criterion(disc_fake_for_gen, real_labels)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % log_every == 0:
                train_loader_with_progress.set_postfix({
                    "Batch": f"{batch_number}/{len(dataloader)}",
                    "D loss": f"{loss_disc.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}",
                })
                datalogs.append({
                    "epoch": epoch + batch_number / len(dataloader),
                    "batch": batch_number / len(dataloader),
                    "D_loss": loss_disc.item(),
                    "G_loss": loss_gen.item(),
                })

        # Generate and save a sample image grid from the fixed noise vector
        generator.eval()
        with torch.no_grad():
            fake_samples = generator(fixed_noise).detach().cpu()
        generator.train()

        grid = make_grid(fake_samples, normalize=True)
        sample_path = os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png")
        save_image(grid, sample_path)

        # Save checkpoints for both networks
        save_checkpoint(
            generator, opt_gen, epoch + 1, loss_gen.item(), accuracy=0.0,
            checkpoint_dir=f"{checkpoint_dir}/generator"
        )
        save_checkpoint(
            discriminator, opt_disc, epoch + 1, loss_disc.item(), accuracy=0.0,
            checkpoint_dir=f"{checkpoint_dir}/discriminator"
        )

        print(f"Epoch {epoch+1}: D loss={loss_disc.item():.4f}, G loss={loss_gen.item():.4f}")
        print(f"Sample grid saved: {sample_path}")

    print("Finished Training")
    return generator, discriminator, datalogs
