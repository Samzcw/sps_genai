import copy
import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


# --- Noise schedules -------------------------------------------------------
# Each takes diffusion_times (values in [0, 1], where 0 = no noise / clean
# image, 1 = pure noise) and returns (noise_rates, signal_rates) such that
# noisy_image = signal_rate * image + noise_rate * noise

def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(diffusion_times * torch.pi / 2)
    noise_rates = torch.sin(diffusion_times * torch.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """
    The schedule actually used for training below. Unlike the plain cosine
    schedule, this keeps signal/noise rates from ever reaching pure 0 or 1,
    which avoids some numerical edge cases at the very start/end of the
    diffusion process.
    """
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


# --- DiffusionModel wrapper --------------------------------------------------

class DiffusionModel(nn.Module):
    """
    Wraps a U-Net (e.g. HW4_DIFFUSION) with:
      - an EMA (exponential moving average) copy of the weights, used for
        validation and generation instead of the raw training weights --
        EMA weights tend to produce noticeably cleaner samples than the
        noisier weights mid-training
      - learned per-channel normalization statistics (mean/std computed
        from the training data), since diffusion works best when images
        are roughly standard-normal rather than raw [0, 1] pixel values
    """
    def __init__(self, model, schedule_fn, ema_decay=0.999):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = current_images
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn(
                (num_images, self.network.num_channels, image_size, image_size),
                device=next(self.parameters()).device
            )
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def _update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        optimizer.step()

        # Update the EMA network -- without this, self.ema_network would
        # stay frozen at its random initial weights forever, and every
        # validation check / generated sample would be using an untrained
        # network regardless of how well self.network trains.
        self._update_ema()

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()


# --- Checkpointing ------------------------------------------------------
# Diffusion checkpoints need to preserve the EMA weights and normalizer
# stats alongside the main network -- the shared save_checkpoint/
# load_checkpoint in checkpoints.py doesn't support those extra fields, so
# this model uses its own save/load pair instead.

def save_diffusion_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='hw4_checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.network.state_dict(),
        'ema_model_state_dict': model.ema_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'normalizer_mean': model.normalizer_mean,
        'normalizer_std': model.normalizer_std,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_diffusion_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.normalizer_mean = checkpoint['normalizer_mean']
    model.normalizer_std = checkpoint['normalizer_std']

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

    return checkpoint['epoch']


# --- Training loop --------------------------------------------------------

def train_diffusion(model, train_loader, val_loader, optimizer, loss_fn, epochs=50, device='cuda',
                     image_size=32, diffusion_steps=20, num_sample_imgs=8,
                     checkpoint_dir='hw4_checkpoints', sample_dir='hw4_samples_diffusion'):
    os.makedirs(sample_dir, exist_ok=True)
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []
        loader_with_progress = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, loss_fn)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')

        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        for images, _ in tqdm(val_loader, ncols=120, desc=f"Epoch {epoch+1} [Val]"):
            images = images.to(device)
            loss = model.test_step(images, loss_fn)
            val_losses.append(loss)
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Generate and save a sample image grid using the EMA network
        generated = model.generate(num_images=num_sample_imgs, diffusion_steps=diffusion_steps, image_size=image_size)
        grid = make_grid(generated.cpu(), normalize=False)
        sample_path = os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png")
        save_image(grid, sample_path)

        checkpoint_path = save_diffusion_checkpoint(
            model, optimizer, epoch + 1, avg_train_loss, avg_val_loss, checkpoint_dir=checkpoint_dir
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = save_diffusion_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss, avg_val_loss,
                checkpoint_dir=os.path.join(checkpoint_dir, 'best')
            )
            print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")
            print(f"Best model path: {best_path}")

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Sample grid saved: {sample_path}")

    print("Finished Training")
