# ALL CODE


import argparse
import copy
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. Data Loader Class 

class DiffusionDataset(Dataset):
    """
    Custom Dataset for loading a subset of animal images for diffusion training.
    Normalises to [-1, 1] to match the range of the Gaussian noise injected
    during the forward diffusion process.
    """

    def __init__(self, data_path, image_size=64, num_images_per_class=120,
                 classes=None, transform=None):
        self.data_path = data_path
        self.image_size = image_size
        self.num_images_per_class = num_images_per_class

        self.all_classes = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])

        self.classes = classes if classes is not None else self.all_classes[:5]
        print(f"Selected classes: {self.classes}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                selected_images = images[:num_images_per_class]
                for img_name in selected_images:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"Total images loaded: {len(self.image_paths)}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


# 2. Forward Diffusion Process  

class DiffusionForwardProcess:
    """
    Forward diffusion: adds noise via the closed-form reparameterisation
        x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon

    This is Algorithm 1 step 4 from the DDPM paper (Ho et al. 2020).
    We NEVER apply noise directly to x_0; all noise is injected through
    pre-computed alpha_bar coefficients so any timestep t can be sampled in
    one shot without running t sequential steps.
    """

    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def add_noise(self, x_0, t):
        """
        Closed-form forward diffusion at timestep t.
        x_0 : [B, C, H, W]
        t   : [B]  long tensor of timestep indices
        Returns (x_t, epsilon) where epsilon is the noise actually added.
        """
        epsilon = torch.randn_like(x_0)

        sqrt_ab  = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        x_t = sqrt_ab * x_0 + sqrt_1ab * epsilon
        return x_t, epsilon

    def get_noisy_images_at_steps(self, x_0, steps):
        noisy_images = []
        for t in steps:
            x_t, _ = self.add_noise(x_0, torch.tensor([t], device=self.device))
            noisy_images.append(x_t)
        return noisy_images


# 3. Model Architecture  

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings (Vaswani et al.).
    Maps scalar timestep t to a fixed-size vector so the U-Net knows
    which noise level it is denoising at.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU activations.
    GroupNorm is preferred over BatchNorm for diffusion models because batch
    sizes are small; SiLU (sigmoid-linear unit) gives smooth gradients.
    Timestep embedding is injected additively after the first conv so each
    block knows what noise level it is operating at.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act   = nn.SiLU()
        self.time_proj = (nn.Linear(time_emb_dim, out_channels)
                          if time_emb_dim is not None else None)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    U-Net denoiser for DDPM.

    Encoder path : ResBlocks + strided Conv2d downsampling.
    Decoder path : ResBlocks + ConvTranspose2d upsampling + skip connections.
    Skip connections preserve spatial detail lost during downsampling.

    Activation choices:
      - SiLU throughout hidden layers  (smooth, non-zero gradient everywhere)
      - No activation on final output  (noise prediction is unbounded)
    Normalisation:
      - GroupNorm (groups=8) instead of BatchNorm because batch sizes are tiny.
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32,
                 time_emb_dim=256, num_res_blocks=2, channel_mults=(1, 2, 4)):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        #  Time embedding pipeline 
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        #  Encoder 
        self.encoder_levels    = nn.ModuleList()
        self.downsamplers      = nn.ModuleList()
        self._enc_out_channels = []

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level  = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.append(ResidualBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
            self.encoder_levels.append(level)
            self._enc_out_channels.append(ch)

            if i < len(channel_mults) - 1:
                self.downsamplers.append(
                    nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.SiLU())
                )
            else:
                self.downsamplers.append(nn.Identity())

        #  Bottleneck 
        self.bottleneck1 = ResidualBlock(ch, ch, time_emb_dim)
        self.bottleneck2 = ResidualBlock(ch, ch, time_emb_dim)

        #  Decoder 
        self.decoder_levels = nn.ModuleList()
        self.upsamplers     = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch  = base_channels * mult
            skip_ch = self._enc_out_channels[-(i + 1)]
            level   = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                in_ch = (ch + skip_ch) if j == 0 else out_ch
                level.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                ch = out_ch
            self.decoder_levels.append(level)

            if i < len(channel_mults) - 1:
                self.upsamplers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
                        nn.SiLU()
                    )
                )
            else:
                self.upsamplers.append(nn.Identity())

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))
        h = self.init_conv(x)

        # Encoder — store one skip per level
        skips = []
        for level_blocks, downsampler in zip(self.encoder_levels, self.downsamplers):
            for block in level_blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsampler(h)

        # Bottleneck
        h = self.bottleneck1(h, t_emb)
        h = self.bottleneck2(h, t_emb)

        # Decoder — concatenate matching skip
        for level_blocks, upsampler in zip(self.decoder_levels, self.upsamplers):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                h = block(h, t_emb)
            h = upsampler(h)

        return self.final_conv(h)


# 4. Custom Loss Function  

def custom_diffusion_loss(noise_pred, noise_true):
    """
    Custom combined L1 + L2 loss for DDPM training.
    L2 (MSE) : penalises large prediction errors heavily -> fast convergence.
    L1 (MAE) : treats all errors equally -> more robust to outlier noise samples.
    Combining both balances speed of convergence with robustness.
    """
    l1 = torch.mean(torch.abs(noise_pred - noise_true))
    l2 = torch.mean((noise_pred - noise_true) ** 2)
    return l1 + l2


# EMA Helper

def update_ema(ema_model, model, decay=0.995):
    """Exponential Moving Average update — stabilises generation quality."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)


# Diffusion Model (forward + reverse combined)

class DiffusionModel:
    def __init__(self, model, forward_process, device='cpu'):
        self.model           = model
        self.forward_process = forward_process
        self.device          = device
        self.T               = forward_process.T

    def predict_noise(self, x_t, t):
        return self.model(x_t, t)

    def denoise_step(self, x_t, t, noise_pred):
        """Single DDPM reverse step."""
        beta              = self.forward_process.betas[t].view(-1, 1, 1, 1)
        alpha             = self.forward_process.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.forward_process.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        x_prev = (1.0 / torch.sqrt(alpha)) * (
            x_t - (1 - alpha) / sqrt_one_minus_ab * noise_pred
        )

        sigma_t = torch.sqrt(beta)
        mask    = (t > 0).float().view(-1, 1, 1, 1)
        x_prev  = x_prev + sigma_t * torch.randn_like(x_t) * mask

        return x_prev

    def sample(self, num_samples, image_size=(3, 64, 64), show_progress=True):
        """Generate new samples starting from pure Gaussian noise."""
        x = torch.randn(num_samples, *image_size, device=self.device)
        samples  = []
        iterator = (tqdm(range(self.T - 1, -1, -1), desc="Sampling")
                    if show_progress else range(self.T - 1, -1, -1))

        for t in iterator:
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = self.predict_noise(x, t_tensor)
            x = self.denoise_step(x, t_tensor, noise_pred)
            if t % 100 == 0:
                samples.append(x.clone())

        return x, samples


# Training Function

def train_diffusion_model(model, ema_model, diffusion, train_loader,
                          epochs=600, lr=1e-4, device='cpu',
                          save_path='models/', print_every=10,
                          patience=40, min_delta=1e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    os.makedirs(save_path, exist_ok=True)

    losses = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    print(f"Starting training for {epochs} epochs on {device}")

    for epoch in range(epochs):
        epoch_loss  = 0
        batch_count = 0

        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images     = images.to(device)
            batch_size = images.size(0)

            t               = torch.randint(0, diffusion.T, (batch_size,), device=device)
            x_t, noise_true = diffusion.forward_process.add_noise(images, t)
            noise_pred      = diffusion.predict_noise(x_t, t)

            loss = custom_diffusion_loss(noise_pred, noise_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA after every batch
            update_ema(ema_model, model)

            epoch_loss  += loss.item()
            batch_count += 1

            if batch_idx % print_every == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        scheduler.step()

        # Early stopping + save best model
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(save_path, 'best_model.pth'))
        else:
            epochs_without_improvement += 1

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"No improve: {epochs_without_improvement}/{patience}")

        # Visualise samples every 20 epochs
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                samples, _ = diffusion.sample(4, show_progress=False)
                visualize_samples(samples, epoch, save_path)

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} — "
                  f"no improvement for {patience} epochs.")
            break

    return losses


# Visualisation Helpers

def visualize_samples(images, epoch=None, save_path='models/'):
    if isinstance(images, torch.Tensor):
        images = images.cpu()
        # Min-max normalize for visibility
        images = images - images.min()
        images = images / (images.max() + 1e-8)
        images = torch.clamp(images, 0, 1).permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(1, min(4, len(images)), figsize=(12, 3))
    if len(images) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
    plt.tight_layout()
    fname = f'samples_epoch_{epoch+1}.png' if epoch is not None else 'samples.png'
    plt.savefig(os.path.join(save_path, fname), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_forward_process(forward_process, image, device, save_path='models/'):
    """Visualise the forward diffusion process (adding noise step by step)."""
    image = image.unsqueeze(0).to(device)
    steps = [0, 50, 100, 200, 400, 600, 800, 999]
    noisy_images = []
    with torch.no_grad():
        for t in steps:
            x_t, _ = forward_process.add_noise(image, torch.tensor([t], device=device))
            noisy_images.append(x_t.cpu())
    fig, axes = plt.subplots(1, len(steps), figsize=(16, 3))
    for ax, img, t in zip(axes, noisy_images, steps):
        img = torch.clamp((img[0] + 1) / 2, 0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f't={t}', fontsize=12)
        ax.axis('off')
    plt.suptitle('Forward Diffusion Process (Adding Noise)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'forward_process.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_loss_graph(losses, save_path='models/'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'loss_graph.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# 5. Test Function  (10 marks)

def test_model(model_path, diffusion, device, num_samples=4, save_path='models/'):
    """Load trained EMA model and generate images from pure Gaussian noise."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint['ema_model_state_dict'])
    diffusion.model.eval()
    print(f"Loaded epoch {checkpoint['epoch']+1}, loss {checkpoint['loss']:.4f}")

    with torch.no_grad():
        samples, sample_steps = diffusion.sample(num_samples, show_progress=True)
        visualize_samples(samples, save_path=save_path)

        if sample_steps:
            fig, axes = plt.subplots(len(sample_steps), num_samples,
                                     figsize=(num_samples * 3, len(sample_steps) * 3))
            for i, s in enumerate(sample_steps):
                for j in range(min(num_samples, s.size(0))):
                    img = s[j].cpu().float()
                    img = img - img.min()
                    img = img / (img.max() + 1e-8)
                    axes[i, j].imshow(img.permute(1, 2, 0).numpy())
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_ylabel(f't={diffusion.T - i*100 - 1}', fontsize=10)
            plt.suptitle('Reverse Diffusion (Noise -> Image)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'sampling_process.png'), dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()

    return samples


# Main

def main():
    parser = argparse.ArgumentParser(description='DDPM Training - MSDS25012')
    parser.add_argument('--data_path',        type=str,   default='data/',   help='Path to animal dataset')
    parser.add_argument('--save_path',        type=str,   default='models/', help='Where to save models/plots')
    parser.add_argument('--epochs',           type=int,   default=600)
    parser.add_argument('--batch_size',       type=int,   default=16)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--image_size',       type=int,   default=64)
    parser.add_argument('--T',                type=int,   default=1000)
    parser.add_argument('--images_per_class', type=int,   default=120)
    args = parser.parse_args()

    #  Dataset 
    print("Loading dataset...")
    dataset = DiffusionDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        num_images_per_class=args.images_per_class
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)} images | Batches: {len(train_loader)}")

    #  Forward process 
    print("Initialising forward process...")
    forward_process = DiffusionForwardProcess(T=args.T, device=device)

    sample_img, _ = dataset[0]
    visualize_forward_process(forward_process, sample_img, device=device, save_path=args.save_path)

    #  Model 
    print("Creating U-Net model...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_emb_dim=256,
        num_res_blocks=2
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    #  EMA model 
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    print("EMA model initialised.")

    diffusion = DiffusionModel(model, forward_process, device)

    #  Train 
    print("Starting training...")
    losses = train_diffusion_model(
        model=model,
        ema_model=ema_model,
        diffusion=diffusion,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_path
    )
    plot_loss_graph(losses, save_path=args.save_path)

    #  Test 
    print("Testing model...")
    # Point diffusion at the EMA model for inference
    diffusion_ema = DiffusionModel(ema_model, forward_process, device)
    test_model(
        model_path=os.path.join(args.save_path, 'best_model.pth'),
        diffusion=diffusion_ema,
        device=device,
        num_samples=4,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
