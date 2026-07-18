import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from .checkpoints import save_checkpoint


def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Run Langevin dynamics (MCMC sampling): starting from inp_imgs (random
    noise or buffered samples), repeatedly nudge every pixel a little in the
    direction that *decreases* the model's energy, plus a small amount of
    injected noise to keep the chain from collapsing onto a single point.

    Uses torch.autograd.grad() rather than .backward() so the gradient is
    computed only w.r.t. inp_imgs -- model parameters are never touched,
    so there's no need to freeze/unfreeze requires_grad on them.
    """
    nn_energy_model.eval()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        energy = nn_energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    nn_energy_model.train()
    return inp_imgs.detach()


class Buffer:
    """
    Replay buffer of previously generated samples. Reusing old samples
    (instead of always starting Langevin dynamics from pure noise) makes
    training much more stable -- this is "persistent contrastive divergence".
    """
    def __init__(self, model, device, img_shape=(3, 32, 32), buffer_size=128, max_len=8192):
        super().__init__()
        self.model = model
        self.device = device
        self.img_shape = img_shape
        self.buffer_size = buffer_size
        self.max_len = max_len
        self.examples = [
            torch.rand((1,) + img_shape, device=self.device) * 2 - 1
            for _ in range(self.buffer_size)
        ]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.buffer_size, 0.05)

        new_rand_imgs = torch.rand((n_new,) + self.img_shape, device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.buffer_size - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)

        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]

        return new_imgs


class Metric:
    """Running average of a scalar value over an epoch."""
    def __init__(self):
        self.reset()

    def update(self, val):
        self.total += val.item()
        self.count += 1

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class EBM(nn.Module):
    def __init__(self, model, alpha, steps, step_size, noise, device, img_shape=(3, 32, 32)):
        super().__init__()
        self.device = device
        self.model = model
        self.buffer = Buffer(self.model, device=device, img_shape=img_shape)

        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                  self.real_out_metric, self.fake_out_metric]:
            m.reset()

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = self.buffer.sample_new_exmps(
            steps=self.steps, step_size=self.step_size, noise=self.noise
        )

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)

        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return self.metrics()

    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand((batch_size,) + self.buffer.img_shape, device=self.device) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

        with torch.no_grad():
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
            cdiv = real_out.mean() - fake_out.mean()

        self.cdiv_loss_metric.update(cdiv)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return {
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }


def train_energy_model(model, train_loader, val_loader=None, device='cpu', epochs=10, lr=1e-4,
                        alpha=0.1, steps=60, step_size=10, noise=0.005, img_shape=(3, 32, 32),
                        checkpoint_dir='hw4_checkpoints', sample_dir='hw4_samples', num_sample_imgs=8):
    """
    Train loop wrapping the EBM class above, following the same shape as
    hw2's train_model / hw3's train_gan: per-epoch progress bar, per-epoch
    checkpoint + sample grid, returns a list of per-epoch metric dicts.
    """
    datalogs = []
    model = model.to(device)
    ebm = EBM(model, alpha=alpha, steps=steps, step_size=step_size, noise=noise,
              device=device, img_shape=img_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))

    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(epochs):
        ebm.reset_metrics()
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for real_imgs, _ in train_loader_with_progress:
            real_imgs = real_imgs.to(device)
            metrics = ebm.train_step(real_imgs, optimizer)
            train_loader_with_progress.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        log_entry = {"epoch": epoch + 1, **metrics}

        if val_loader is not None:
            ebm.reset_metrics()
            for real_imgs, _ in val_loader:
                real_imgs = real_imgs.to(device)
                val_metrics = ebm.test_step(real_imgs)
            log_entry.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            print("Validation - " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
        else:
            print(f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        datalogs.append(log_entry)

        # Save a grid of the most recent buffer samples (recycled from
        # training, no extra Langevin dynamics needed)
        sample_imgs = torch.cat(ebm.buffer.examples[-num_sample_imgs:])
        grid = make_grid(sample_imgs.cpu(), normalize=True, value_range=(-1, 1))
        sample_path = os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png")
        save_image(grid, sample_path)
        print(f"Sample grid saved: {sample_path}")

        save_checkpoint(
            model, optimizer, epoch + 1, log_entry["loss"], accuracy=0.0,
            checkpoint_dir=f"{checkpoint_dir}/energy"
        )

    print("Finished Training")
    return model, datalogs