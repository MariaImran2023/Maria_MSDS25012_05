import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Data Loader Class (5 marks) ====================

class DiffusionDataset(Dataset):
    """
    Custom Dataset class for loading images for diffusion model training
    """
    def __init__(self, data_path, image_size=64, num_images_per_class=20, 
                 classes=None, transform=None):
        """
        Args:
            data_path: Path to the dataset containing class folders
            image_size: Size to resize images to (default: 64x64)
            num_images_per_class: Number of images to select from each class
            classes: List of class names to use (if None, use first 5 classes)
            transform: Optional transforms to apply
        """
        self.data_path = data_path
        self.image_size = image_size
        self.num_images_per_class = num_images_per_class
        
        # Get all classes from the dataset
        self.all_classes = [d for d in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, d))]
        self.all_classes.sort()
        
        # Select classes
        if classes is None:
            # Use first 5 classes if not specified
            self.classes = self.all_classes[:5]
        else:
            self.classes = classes
            
        print(f"Selected classes: {self.classes}")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect image paths
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                # Select specified number of images
                selected_images = images[:num_images_per_class]
                
                for img_name in selected_images:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total images loaded: {len(self.image_paths)}")
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
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
            # Return a random image as fallback
            return self.__getitem__(random.randint(0, len(self)-1))

# ==================== Forward Process (10 marks) ====================

class DiffusionForwardProcess:
    """
    Handles the forward diffusion process of adding noise to images
    """
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        """
        Args:
            T: Number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            device: Device to run computations on
        """
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        
        # Calculate alphas and alpha bars
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Pre-calculate coefficients for sampling
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def add_noise(self, x_0, t):
        """
        Add noise to clean image at step t
        x_0: Clean image tensor [batch_size, channels, height, width]
        t: Timestep (can be a tensor of timesteps)
        
        Returns: Noisy image at timestep t
        """
        # Create noise
        noise = torch.randn_like(x_0)
        
        # Get coefficients for this timestep
        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
        
        # Reshape for broadcasting
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def get_noisy_images_at_steps(self, x_0, steps):
        """
        Get noisy images at specified steps for visualization
        """
        noisy_images = []
        for t in steps:
            x_t, _ = self.add_noise(x_0, torch.tensor([t], device=self.device))
            noisy_images.append(x_t)
        return noisy_images

# ==================== Model Architecture (10 marks) ====================

class SinusoidalPositionEmbedding(nn.Module):
    """Positional embeddings for timesteps using sinusoidal functions"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class ResidualBlock(nn.Module):
    """Residual block with optional timestep embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
        # Time embedding projection
        self.time_emb = None
        if time_emb_dim is not None:
            self.time_emb = nn.Linear(time_emb_dim, out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb=None):
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding if present
        if self.time_emb is not None and t_emb is not None:
            t_emb_out = self.time_emb(t_emb)
            t_emb_out = t_emb_out[:, :, None, None]  # Add spatial dimensions
            h = h + t_emb_out
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        # Skip connection
        skip = self.skip(x)
        
        return h + skip

class UNet(nn.Module):
    """
    U-Net architecture for denoising diffusion models
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 time_emb_dim=256, num_res_blocks=2, channel_mults=(1, 2, 4)):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # Residual blocks
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(current_channels, out_channels, time_emb_dim)
                )
                current_channels = out_channels
            
            # Downsampling (except for last block)
            if i < len(channel_mults) - 1:
                self.encoder_downsample.append(
                    nn.Sequential(
                        nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1),
                        nn.SiLU()
                    )
                )
            else:
                self.encoder_downsample.append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(current_channels, current_channels, time_emb_dim),
            ResidualBlock(current_channels, current_channels, time_emb_dim)
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            
            # Residual blocks with skip connections
            for _ in range(num_res_blocks + 1):
                self.decoder_blocks.append(
                    ResidualBlock(current_channels + out_channels, out_channels, time_emb_dim)
                )
                current_channels = out_channels
            
            # Upsampling (except for last block)
            if i < len(channel_mults) - 1:
                self.decoder_upsample.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(current_channels, current_channels, 
                                         kernel_size=4, stride=2, padding=1),
                        nn.SiLU()
                    )
                )
            else:
                self.decoder_upsample.append(nn.Identity())
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        skip_connections = []
        block_idx = 0
        
        for i, (block, downsample) in enumerate(zip(self.encoder_blocks, self.encoder_downsample)):
            # Residual blocks
            for _ in range(2 if i == 0 else 1):  # First block has 2 residual blocks
                if block_idx < len(self.encoder_blocks):
                    h = self.encoder_blocks[block_idx](h, t_emb)
                    skip_connections.append(h)
                    block_idx += 1
            h = downsample(h)
        
        # Bottleneck
        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h, t_emb)
        
        # Decoder
        block_idx = 0
        for i, (block, upsample) in enumerate(zip(self.decoder_blocks, self.decoder_upsample)):
            # Residual blocks with skip connections
            for _ in range(2 if i == 0 else 1):
                if block_idx < len(self.decoder_blocks):
                    # Concatenate with skip connection
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = self.decoder_blocks[block_idx](h, t_emb)
                    block_idx += 1
            h = upsample(h)
        
        # Final convolution
        h = self.final_conv(h)
        
        return h

# ==================== Diffusion Model ====================

class DiffusionModel:
    """
    Main Diffusion Model class combining forward and reverse processes
    """
    def __init__(self, model, forward_process, device='cpu'):
        self.model = model
        self.forward_process = forward_process
        self.device = device
        self.T = forward_process.T
        
    def predict_noise(self, x_t, t):
        """Predict noise from noisy image"""
        return self.model(x_t, t)
    
    def denoise_step(self, x_t, t, noise_pred):
        """Single denoising step"""
        # Calculate coefficients
        beta = self.forward_process.betas[t]
        alpha = self.forward_process.alphas[t]
        alpha_bar = self.forward_process.alpha_bars[t]
        sqrt_alpha_bar = self.forward_process.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.forward_process.sqrt_one_minus_alpha_bars[t]
        
        # Denoising formula
        if t > 0:
            # Add some noise for stochastic sampling
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta)
        else:
            z = torch.zeros_like(x_t)
            sigma_t = 0
        
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * noise_pred) + sigma_t * z
        x_prev = (1 / torch.sqrt(alpha)) * (
            x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * noise_pred
        ) + sigma_t * z
        
        return x_prev
    
    def sample(self, num_samples, image_size=(3, 64, 64), show_progress=True):
        """
        Generate new samples from noise
        """
        # Start from pure noise
        x = torch.randn(num_samples, *image_size, device=self.device)
        
        samples = []
        iterator = tqdm(range(self.T-1, -1, -1), desc="Sampling") if show_progress else range(self.T-1, -1, -1)
        
        for t in iterator:
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.predict_noise(x, t_tensor)
            
            # Denoise step
            x = self.denoise_step(x, t_tensor, noise_pred)
            
            if t % 100 == 0 and show_progress:
                samples.append(x.clone())
        
        return x, samples

# ==================== Training Function ====================

def train_diffusion_model(model, diffusion, train_loader, epochs=100, lr=1e-4, 
                          device='cpu', save_path='models/', print_every=10):
    """
    Train the diffusion model
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Custom L2 loss (5 marks)
    def custom_loss(noise_pred, noise_true):
        """Custom loss function combining L1 and L2 losses"""
        l1_loss = torch.mean(torch.abs(noise_pred - noise_true))
        l2_loss = torch.mean((noise_pred - noise_true) ** 2)
        return l1_loss + l2_loss  # Combined loss
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    losses = []
    best_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs")
    print(f"Device: {device}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.T, (batch_size,), device=device)
            
            # Add noise to images
            x_t, noise_true = diffusion.forward_process.add_noise(images, t)
            
            # Predict noise
            noise_pred = diffusion.predict_noise(x_t, t)
            
            # Calculate loss
            loss = custom_loss(noise_pred, noise_true)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % print_every == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Visualize samples every 20 epochs
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                samples, _ = diffusion.sample(4, show_progress=False)
                visualize_samples(samples, epoch, save_path)
    
    return losses

# ==================== Visualization Functions ====================

def visualize_samples(images, epoch=None, save_path='models/'):
    """Visualize generated samples"""
    if isinstance(images, torch.Tensor):
        images = images.cpu()
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        images = images.permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(1, min(4, len(images)), figsize=(12, 3))
    if len(images) == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
    
    plt.tight_layout()
    
    if epoch is not None:
        plt.savefig(os.path.join(save_path, f'samples_epoch_{epoch+1}.png'), dpi=150, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_path, 'samples.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_forward_process(diffusion, image, save_path='models/'):
    """Visualize the forward diffusion process"""
    image = image.unsqueeze(0).to(device)
    
    # Steps to visualize
    steps = [0, 50, 100, 200, 400, 600, 800, 999]
    
    noisy_images = []
    with torch.no_grad():
        for t in steps:
            x_t, _ = diffusion.forward_process.add_noise(image, torch.tensor([t], device=device))
            noisy_images.append(x_t.cpu())
    
    # Denormalize
    fig, axes = plt.subplots(1, len(steps), figsize=(16, 3))
    
    for i, (ax, t) in enumerate(zip(axes, steps)):
        img = noisy_images[i][0]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        ax.set_title(f't={t}', fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Forward Diffusion Process (Adding Noise)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'forward_process.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_loss_graph(losses, save_path='models/'):
    """Plot training loss graph"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'loss_graph.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# ==================== Test Function (10 marks) ====================

def test_model(model_path, diffusion, device, num_samples=4):
    """
    Test function to generate images from noise using trained model
    """
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint['model_state_dict'])
    diffusion.model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.4f}")
    
    # Generate samples
    with torch.no_grad():
        print("Generating samples...")
        samples, sample_steps = diffusion.sample(num_samples, show_progress=True)
        
        # Visualize samples
        visualize_samples(samples, save_path='models/')
        
        # Visualize sampling process
        if sample_steps:
            fig, axes = plt.subplots(len(sample_steps), num_samples, figsize=(num_samples*3, len(sample_steps)*3))
            for i, samples_at_t in enumerate(sample_steps):
                for j in range(min(num_samples, samples_at_t.size(0))):
                    img = samples_at_t[j].cpu()
                    img = (img + 1) / 2
                    img = torch.clamp(img, 0, 1)
                    img = img.permute(1, 2, 0).numpy()
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_ylabel(f't={diffusion.T - (i*100) - 1}', fontsize=10)
            plt.suptitle('Sampling Process (Reverse Diffusion)', fontsize=14)
            plt.tight_layout()
            plt.savefig('models/sampling_process.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
    
    return samples

# ==================== Main Execution ====================

def main():
    # Configuration
    DATA_PATH = 'data/'  # Update with your dataset path
    IMAGE_SIZE = 64
    T = 1000
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5
    IMAGES_PER_CLASS = 20
    
    # 1. Create Data Loader (5 marks)
    print("Loading dataset...")
    dataset = DiffusionDataset(
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
        num_images_per_class=IMAGES_PER_CLASS
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of batches: {len(train_loader)}")
    
    # 2. Initialize forward process (10 marks)
    print("Initializing forward process...")
    forward_process = DiffusionForwardProcess(T=T, device=device)
    
    # Visualize forward process with a sample image
    sample_img, _ = dataset[0]
    visualize_forward_process(forward_process, sample_img)
    
    # 3. Create model (10 marks)
    print("Creating U-Net model...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
        num_res_blocks=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Create diffusion model
    diffusion = DiffusionModel(model, forward_process, device)
    
    # 5. Train the model
    print("Starting training...")
    losses = train_diffusion_model(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_path='models/'
    )
    
    # Plot loss graph (5 marks)
    plot_loss_graph(losses)
    
    # 6. Test the model (10 marks)
    print("Testing the model...")
    test_model('models/best_model.pth', diffusion, device, num_samples=4)
    
    print("Assignment completed successfully!")

if __name__ == "__main__":
    main()