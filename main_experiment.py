"""
Comparative Analysis of Generative Models: Diffusion vs GAN vs Transformer
===========================================================================
This script trains and evaluates three generative architectures on Fashion-MNIST:
  1. DDPM  (Denoising Diffusion Probabilistic Model)
  2. DCGAN (Deep Convolutional GAN)
  3. Image Transformer (Autoregressive, GPT-style)

Results (loss curves + generated image grids) are saved to ./results/
"""

import os
import time
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
DATASET = "FashionMNIST"  # lightweight 28x28 grayscale
BATCH_SIZE = 128
IMAGE_SIZE = 28
CHANNELS = 1
LATENT_DIM = 100  # for GAN
NUM_EPOCHS_GAN = 5
NUM_EPOCHS_DDPM = 5
NUM_EPOCHS_TRANSFORMER = 5
LR = 2e-4

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")


# ============================= DATA =========================================
def get_dataloader():
    """Return a DataLoader for Fashion-MNIST normalised to [-1, 1]."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ]
    )
    dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ============================= 1. DCGAN =====================================
class DCGANGenerator(nn.Module):
    """Transposed-conv generator: z -> 1x28x28 image."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, CHANNELS, 3, 1, 1),  # -> 1x28x28
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 7, 7)
        return self.net(x)


class DCGANDiscriminator(nn.Module):
    """Conv discriminator: 1x28x28 -> real/fake scalar."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1),  # -> 64x14x14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128x7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1),  # -> 256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_dcgan(dataloader):
    """Train DCGAN and return loss history + timing info."""
    print("\n" + "=" * 60)
    print("Training DCGAN")
    print("=" * 60)
    G = DCGANGenerator().to(DEVICE)
    D = DCGANDiscriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    g_losses, d_losses = [], []
    start = time.time()

    for epoch in range(NUM_EPOCHS_GAN):
        epoch_g, epoch_d, n = 0.0, 0.0, 0
        pbar = tqdm(dataloader, desc=f"DCGAN Epoch {epoch+1}/{NUM_EPOCHS_GAN}")
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(DEVICE)
            bs = real_imgs.size(0)
            real_label = torch.ones(bs, 1, device=DEVICE)
            fake_label = torch.zeros(bs, 1, device=DEVICE)

            # --- Discriminator ---
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z).detach()
            loss_D = criterion(D(real_imgs), real_label) + criterion(
                D(fake_imgs), fake_label
            )
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator ---
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z)
            loss_G = criterion(D(fake_imgs), real_label)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()
            n += 1
            pbar.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}")

        g_losses.append(epoch_g / n)
        d_losses.append(epoch_d / n)
        print(
            f"  Epoch {epoch+1}: G_loss={g_losses[-1]:.4f}  D_loss={d_losses[-1]:.4f}"
        )

    train_time = time.time() - start

    # Inference timing
    G.eval()
    with torch.no_grad():
        t0 = time.time()
        z = torch.randn(64, LATENT_DIM, device=DEVICE)
        samples = G(z)
        infer_time = time.time() - t0

    save_grid(samples, "dcgan_samples.png", title="DCGAN Generated Samples")
    plot_losses(
        {"Generator": g_losses, "Discriminator": d_losses},
        "dcgan_loss.png",
        "DCGAN Loss Curves",
    )

    print(f"  Training time : {train_time:.1f}s")
    print(f"  Inference time (64 imgs): {infer_time:.4f}s")
    return {
        "train_time": train_time,
        "infer_time": infer_time,
        "g_losses": g_losses,
        "d_losses": d_losses,
    }


# ============================= 2. DDPM ======================================
# Simple Denoising Diffusion Probabilistic Model
# -----------------------------------------------

TIMESTEPS = 200  # keep small for speed


def linear_beta_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time-embedding injection."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.bn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(self.bn2(self.conv2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """Minimal U-Net for 28x28 images with time conditioning."""

    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )
        # Encoder
        self.enc1 = ResBlock(CHANNELS, 64, time_dim)
        self.enc2 = ResBlock(64, 128, time_dim)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bot = ResBlock(128, 256, time_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ResBlock(256, 128, time_dim)  # concat skip
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ResBlock(128, 64, time_dim)
        self.out = nn.Conv2d(64, CHANNELS, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # Encoder
        e1 = self.enc1(x, t_emb)  # 28x28
        e2 = self.enc2(self.pool(e1), t_emb)  # 14x14
        # Bottleneck
        b = self.bot(self.pool(e2), t_emb)  # 7x7
        # Decoder
        d2 = self.up2(b)  # 14x14
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        d1 = self.up1(d2)  # 28x28
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        return self.out(d1)


class DDPM:
    """Wrapper for forward/reverse diffusion process."""

    def __init__(self, model, timesteps=TIMESTEPS, device=DEVICE):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - torch.cat([torch.tensor([0.0], device=device), alpha_cumprod[:-1]])) / (1.0 - alpha_cumprod)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ac = self.sqrt_alpha_cumprod[t][:, None, None, None]
        s_omac = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        return s_ac * x0 + s_omac * noise, noise

    def p_losses(self, x0):
        """Compute training loss (predict noise)."""
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=self.device)
        noisy, noise = self.q_sample(x0, t)
        pred_noise = self.model(noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n=64):
        """Reverse diffusion: generate n images."""
        self.model.eval()
        x = torch.randn(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)
            beta = self.betas[i]
            sqrt_recip = self.sqrt_recip_alpha[i]
            sqrt_omac = self.sqrt_one_minus_alpha_cumprod[i]
            x = sqrt_recip * (x - beta / sqrt_omac * pred_noise)
            if i > 0:
                x = x + torch.sqrt(self.posterior_variance[i]) * torch.randn_like(x)
        return x.clamp(-1, 1)


def train_ddpm(dataloader):
    """Train DDPM and return loss history + timing info."""
    print("\n" + "=" * 60)
    print("Training DDPM")
    print("=" * 60)
    model = SimpleUNet().to(DEVICE)
    diffusion = DDPM(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    start = time.time()

    for epoch in range(NUM_EPOCHS_DDPM):
        epoch_loss, n = 0.0, 0
        pbar = tqdm(dataloader, desc=f"DDPM  Epoch {epoch+1}/{NUM_EPOCHS_DDPM}")
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE)
            loss = diffusion.p_losses(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(epoch_loss / n)
        print(f"  Epoch {epoch+1}: loss={losses[-1]:.4f}")

    train_time = time.time() - start

    # Inference timing
    t0 = time.time()
    samples = diffusion.sample(64)
    infer_time = time.time() - t0

    save_grid(samples, "ddpm_samples.png", title="DDPM Generated Samples")
    plot_losses({"MSE Loss": losses}, "ddpm_loss.png", "DDPM Loss Curve")

    print(f"  Training time : {train_time:.1f}s")
    print(f"  Inference time (64 imgs): {infer_time:.4f}s")
    return {"train_time": train_time, "infer_time": infer_time, "losses": losses}


# ============================= 3. IMAGE TRANSFORMER =========================
# Autoregressive GPT-style model operating on quantised pixel sequences.
# Pixels are quantised into NUM_PIXEL_BINS discrete levels to form a vocabulary.
# ---------------------------------------------------------------------------

NUM_PIXEL_BINS = 16  # quantise [0,255] -> 16 levels


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ImageTransformer(nn.Module):
    """
    Small autoregressive transformer for 28x28 grayscale images.
    We split each image into patches (4x4) to reduce sequence length
    from 784 to 49 tokens, each patch quantised to one of NUM_PIXEL_BINS^16
    ... that's too many. Instead, we use a hybrid: flatten patches into
    patch embeddings via a linear layer and predict the full patch at once
    via a cross-entropy over binned average pixel value per patch.

    Actually, for simplicity and to truly demonstrate autoregressive generation,
    we'll work with a reduced 14x14 image (downsampled) = 196 tokens,
    each token = one of 16 discrete pixel levels.
    """

    def __init__(
        self,
        vocab_size=NUM_PIXEL_BINS,
        seq_len=196,  # 14*14
        embed_dim=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, seq_len) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """idx: (B, seq_len) of token indices in [0, vocab_size)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, n=64, temperature=1.0):
        """Autoregressively generate n images."""
        self.eval()
        # Start with a BOS-like token (0)
        idx = torch.zeros(n, 1, dtype=torch.long, device=next(self.parameters()).device)
        for _ in range(self.seq_len - 1):
            logits = self.forward(idx)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx  # (n, seq_len) of discrete tokens


def quantise_images(imgs, num_bins=NUM_PIXEL_BINS):
    """Convert [-1,1] images to discrete token sequences of length 14*14=196.
    We downsample 28x28 -> 14x14 via avg pooling, then bin to [0, num_bins-1].
    """
    # imgs: (B, 1, 28, 28) in [-1, 1]
    imgs_down = F.avg_pool2d(imgs, 2)  # (B, 1, 14, 14)
    imgs_01 = (imgs_down + 1.0) / 2.0  # -> [0, 1]
    tokens = (imgs_01 * (num_bins - 1)).long().clamp(0, num_bins - 1)
    tokens = tokens.view(tokens.size(0), -1)  # (B, 196)
    return tokens


def tokens_to_images(tokens, num_bins=NUM_PIXEL_BINS):
    """Convert token sequences back to images (14x14, upsampled to 28x28)."""
    imgs = tokens.float() / (num_bins - 1)  # [0, 1]
    imgs = imgs.view(-1, 1, 14, 14)
    imgs = F.interpolate(imgs, size=(28, 28), mode="nearest")
    imgs = imgs * 2 - 1  # -> [-1, 1]
    return imgs


def train_transformer(dataloader):
    """Train the autoregressive Image Transformer."""
    print("\n" + "=" * 60)
    print("Training Image Transformer")
    print("=" * 60)
    model = ImageTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    start = time.time()

    for epoch in range(NUM_EPOCHS_TRANSFORMER):
        epoch_loss, n = 0.0, 0
        pbar = tqdm(
            dataloader,
            desc=f"Trans Epoch {epoch+1}/{NUM_EPOCHS_TRANSFORMER}",
        )
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE)
            tokens = quantise_images(imgs)  # (B, 196)
            # Teacher forcing: input = tokens[:, :-1], target = tokens[:, 1:]
            inp = tokens[:, :-1]
            target = tokens[:, 1:]
            logits = model(inp)  # (B, 195, vocab_size)
            loss = F.cross_entropy(logits.reshape(-1, NUM_PIXEL_BINS), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(epoch_loss / n)
        print(f"  Epoch {epoch+1}: loss={losses[-1]:.4f}")

    train_time = time.time() - start

    # Inference timing
    model.eval()
    t0 = time.time()
    gen_tokens = model.generate(64, temperature=0.8)
    infer_time = time.time() - t0
    samples = tokens_to_images(gen_tokens)

    save_grid(samples, "transformer_samples.png", title="Transformer Generated Samples")
    plot_losses({"CE Loss": losses}, "transformer_loss.png", "Transformer Loss Curve")

    print(f"  Training time : {train_time:.1f}s")
    print(f"  Inference time (64 imgs): {infer_time:.4f}s")
    return {"train_time": train_time, "infer_time": infer_time, "losses": losses}


# ============================= UTILS ========================================
def save_grid(images, filename, title="", nrow=8):
    """Save a grid of generated images (expected in [-1,1])."""
    images = images.detach().cpu()
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_losses(loss_dict, filename, title):
    """Plot one or more loss curves and save to file."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in loss_dict.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(results):
    """Create a comparison bar chart of training and inference times."""
    models = list(results.keys())
    train_times = [results[m]["train_time"] for m in models]
    infer_times = [results[m]["infer_time"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(models, train_times, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[0].set_title("Training Time (seconds)")
    axes[0].set_ylabel("Time (s)")
    for i, v in enumerate(train_times):
        axes[0].text(i, v + 0.5, f"{v:.1f}s", ha="center", fontweight="bold")

    axes[1].bar(models, infer_times, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[1].set_title("Inference Time – 64 images (seconds)")
    axes[1].set_ylabel("Time (s)")
    for i, v in enumerate(infer_times):
        axes[1].text(i, v + 0.001, f"{v:.4f}s", ha="center", fontweight="bold")

    fig.suptitle("Generative Models Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================= MAIN =========================================
def main():
    print("Loading dataset...")
    dataloader = get_dataloader()
    print(f"Dataset: {DATASET}, batches: {len(dataloader)}, batch_size: {BATCH_SIZE}")

    results = {}

    # 1. DCGAN
    results["DCGAN"] = train_dcgan(dataloader)

    # 2. DDPM
    results["DDPM"] = train_ddpm(dataloader)

    # 3. Transformer
    results["Transformer"] = train_transformer(dataloader)

    # Comparison chart
    plot_comparison(results)

    # Save numeric results to JSON
    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {
            "train_time": v["train_time"],
            "infer_time": v["infer_time"],
        }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(serialisable, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL DONE — results saved to ./results/")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:15s}  train={r['train_time']:.1f}s  infer={r['infer_time']:.4f}s")


if __name__ == "__main__":
    main()
