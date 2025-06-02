# helper/pretrained_diffusion_helper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import scipy.io
from pathlib import Path

class PretrainedDiffusionNoiseGenerator(nn.Module):
    """
    Adapter for pretrained diffusion models to generate camera noise.
    Uses a pretrained UNet and fine-tunes it for noise residual generation.
    """
    
    def __init__(self, 
                 model_name="google/ddpm-celebahq-256",  # Or other pretrained model
                 noise_list='shot_read_uniform_row1_rowt_fixed1',
                 device='cuda:0',
                 lambda_phys=0.1,
                 freeze_backbone=True):
        super(PretrainedDiffusionNoiseGenerator, self).__init__()
        
        self.device = device
        self.lambda_phys = lambda_phys
        self.noise_list = noise_list
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained model
        print(f"Loading pretrained model: {model_name}")
        try:
            # Try loading from pipeline first
            pipeline = DDPMPipeline.from_pretrained(model_name)
            self.unet = pipeline.unet
            self.scheduler = pipeline.scheduler
            print("Loaded from pipeline")
        except:
            # Fallback to direct UNet loading
            self.unet = UNet2DModel.from_pretrained(model_name)
            self.scheduler = DDPMScheduler.from_pretrained(model_name)
            print("Loaded UNet directly")
        
        # Adapt the model for our task
        self._adapt_model()
        
        # Physics parameters (learnable)
        self._init_physics_parameters()
        
        # Fixed pattern noise
        if 'fixed1' in noise_list:
            self._init_fixed_noise()
        
        # Periodic noise parameters
        if 'periodic' in noise_list:
            self.periodic_params = nn.Parameter(
                torch.tensor([0.0050, 0.0050, 0.0050], dtype=torch.float32, device=device) * 100
            )
        
        self.indices = None
        
    def _adapt_model(self):
        """Adapt the pretrained model for our specific task"""
        original_in_channels = self.unet.config.in_channels
        target_in_channels = 8  # r_t(4) + x(4)
        
        # If input channels don't match, we need to adapt
        if original_in_channels != target_in_channels:
            print(f"Adapting input channels from {original_in_channels} to {target_in_channels}")
            
            # Method 1: Add an adapter layer
            self.input_adapter = nn.Conv2d(target_in_channels, original_in_channels, 
                                         kernel_size=1, padding=0)
            
            # Initialize to preserve some pretrained features
            with torch.no_grad():
                # Copy first 3 channels if available (RGB)
                if original_in_channels >= 3 and target_in_channels >= 3:
                    self.input_adapter.weight[:3, :3] = torch.eye(3).unsqueeze(-1).unsqueeze(-1)
                
                # Initialize remaining weights small
                self.input_adapter.weight[3:] *= 0.1
        else:
            self.input_adapter = nn.Identity()
        
        # Adapt output channels if needed
        original_out_channels = self.unet.config.out_channels
        target_out_channels = 4  # RGBN noise residual
        
        if original_out_channels != target_out_channels:
            print(f"Adapting output channels from {original_out_channels} to {target_out_channels}")
            self.output_adapter = nn.Conv2d(original_out_channels, target_out_channels, 
                                          kernel_size=1, padding=0)
            # Initialize small to preserve pretrained representations initially
            self.output_adapter.weight.data *= 0.1
        else:
            self.output_adapter = nn.Identity()
        
        # Optionally freeze the backbone
        if self.freeze_backbone:
            print("Freezing pretrained backbone")
            for param in self.unet.parameters():
                param.requires_grad = False
            
            # Only train the last few layers
            if hasattr(self.unet, 'up_blocks'):
                for param in self.unet.up_blocks[-1].parameters():
                    param.requires_grad = True
            if hasattr(self.unet, 'conv_out'):
                for param in self.unet.conv_out.parameters():
                    param.requires_grad = True
    
    def _init_physics_parameters(self):
        """Initialize learnable physics parameters"""
        self.shot_noise = nn.Parameter(torch.tensor(0.00002*10000, dtype=torch.float32, device=self.device))
        self.read_noise = nn.Parameter(torch.tensor(0.000002*10000, dtype=torch.float32, device=self.device))
        
        if 'row1' in self.noise_list:
            self.row_noise = nn.Parameter(torch.tensor(0.000002*1000, dtype=torch.float32, device=self.device))
        if 'rowt' in self.noise_list:
            self.row_noise_temp = nn.Parameter(torch.tensor(0.000002*1000, dtype=torch.float32, device=self.device))
        if 'uniform' in self.noise_list:
            self.uniform_noise = nn.Parameter(torch.tensor(0.00001*10000, dtype=torch.float32, device=self.device))
    
    def _init_fixed_noise(self):
        """Initialize fixed pattern noise"""
        try:
            _script_dir = Path(__file__).parent
            _root_dir = _script_dir.parent
            mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), 
                                          dtype=torch.float32, device=self.device).unsqueeze(0)
        except:
            print("Warning: Could not load fixed pattern noise, using zeros")
            self.fixednoiset = torch.zeros(1, 4, 640, 1080, device=self.device)
    
    def _get_physics_noise(self, x):
        """Generate physics-based noise components"""
        noise = torch.zeros_like(x)
        
        # Shot and read noise
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            variance = x * self.shot_noise + self.read_noise
            shot_read_noise = torch.randn_like(x) * variance
            noise += shot_read_noise
        elif 'read' in self.noise_list:
            noise += torch.randn_like(x) * self.read_noise
            
        # Uniform noise
        if 'uniform' in self.noise_list:
            uniform_noise = self.uniform_noise * torch.rand_like(x)
            noise += uniform_noise
            
        # Row noise (spatial)
        if 'row1' in self.noise_list:
            row_noise = self.row_noise * torch.randn(*x.shape[:-2], x.shape[-1], device=self.device).unsqueeze(-2)
            noise += row_noise
            
        # Row noise (temporal)  
        if 'rowt' in self.noise_list:
            row_noise_temp = self.row_noise_temp * torch.randn(*x.shape[:-3], x.shape[-1], device=self.device).unsqueeze(-2).unsqueeze(-2)
            noise += row_noise_temp
            
        # Fixed pattern noise
        if 'fixed1' in self.noise_list:
            if self.indices is not None:
                i1, i2 = self.indices[0], self.indices[1]
            else:
                i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
                i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            fixed_noise = self.fixednoiset[..., i1:i1+x.shape[-2], i2:i2+x.shape[-1]]
            noise += fixed_noise
            
        # Periodic noise
        if 'periodic' in self.noise_list:
            periodic_noise = torch.zeros(x.shape, dtype=torch.cfloat, device=self.device)
            periodic_noise[..., 0, 0] = self.periodic_params[0] * torch.randn(x.shape[:2], device=self.device)
            
            periodic0 = self.periodic_params[1] * torch.randn(x.shape[:2], device=self.device)
            periodic1 = self.periodic_params[2] * torch.randn(x.shape[:2], device=self.device)
            
            periodic_noise[..., 0, x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            periodic_noise[..., 0, 3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)
            
            periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise, norm="ortho"))
            noise += periodic_gen
            
        return noise
    
    def forward_diffusion(self, r_0, timesteps):
        """Forward diffusion using pretrained scheduler"""
        noise = torch.randn_like(r_0)
        noisy_residuals = self.scheduler.add_noise(r_0, noise, timesteps)
        return noisy_residuals, noise
    
    def predict_noise(self, r_t, x, timesteps, unconditional_prob=0.1):
        """Predict noise using adapted pretrained UNet"""
        batch_size = r_t.shape[0]
        
        # Classifier-free guidance
        if self.training and torch.rand(1).item() < unconditional_prob:
            x_cond = torch.zeros_like(x)
        else:
            x_cond = x
        
        # Prepare input
        model_input = torch.cat([r_t, x_cond], dim=1)  # Concatenate along channel dim
        
        # Adapt input channels
        model_input = self.input_adapter(model_input)
        
        # Run through pretrained UNet
        noise_pred = self.unet(model_input, timesteps).sample
        
        # Adapt output channels
        noise_pred = self.output_adapter(noise_pred)
        
        return noise_pred
    
    def compute_loss(self, x, phi=None):
        """Compute diffusion loss with physics regularization"""
        batch_size = x.shape[0]
        
        # Sample ground truth residual from physics model
        r_0 = self._get_physics_noise(x)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                (batch_size,), device=self.device)
        
        # Forward diffusion
        r_t, noise = self.forward_diffusion(r_0, timesteps)
        
        # Predict noise
        predicted_noise = self.predict_noise(r_t, x, timesteps)
        
        # Main diffusion loss
        noise_loss = F.mse_loss(predicted_noise, noise)
        
        # Physics regularization (simplified)
        if self.lambda_phys > 0:
            physics_target = self._get_physics_noise(x).detach()
            physics_loss = F.mse_loss(r_0, physics_target)
        else:
            physics_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = noise_loss + self.lambda_phys * physics_loss
        return total_loss, noise_loss, physics_loss
    
    @torch.no_grad()
    def sample(self, x, phi=None, w=1.0, num_inference_steps=50):
        """Sample noise residual using DDPM sampling with fewer steps"""
        batch_size = x.shape[0]
        
        # Set scheduler for inference with fewer steps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Start from random noise
        r_t = torch.randn(batch_size, 4, x.shape[-2], x.shape[-1], device=self.device)
        
        for t in self.scheduler.timesteps:
            t_tensor = t.unsqueeze(0).repeat(batch_size).to(self.device)
            
            # Predict noise
            predicted_noise_cond = self.predict_noise(r_t, x, t_tensor, unconditional_prob=0.0)
            
            # Classifier-free guidance
            if w != 1.0:
                predicted_noise_uncond = self.predict_noise(r_t, torch.zeros_like(x), t_tensor, unconditional_prob=1.0)
                predicted_noise = (1 + w) * predicted_noise_cond - w * predicted_noise_uncond
            else:
                predicted_noise = predicted_noise_cond
            
            # Scheduler step
            r_t = self.scheduler.step(predicted_noise, t, r_t).prev_sample
        
        return r_t
    
    def forward(self, x, phi=None, w=1.0):
        """Generate noisy image: y = x + r"""
        if self.training:
            return self.compute_loss(x, phi)
        else:
            r = self.sample(x, phi, w, num_inference_steps=20)  # Fewer steps for speed
            noisy = x + r
            return torch.clamp(noisy, 0, 1)


def load_pretrained_diffusion_generator(model_name="google/ddpm-celebahq-256", 
                                       noise_list='shot_read_uniform_row1_rowt_fixed1',
                                       device='cuda:0',
                                       checkpoint_path=None,
                                       freeze_backbone=True):
    """
    Load and optionally fine-tune a pretrained diffusion model
    
    Popular pretrained models to try:
    - "google/ddpm-celebahq-256": Good for 256x256 images
    - "google/ddpm-church-256": Alternative for 256x256
    - "google/ddpm-bedroom-256": Another 256x256 option
    - "runwayml/stable-diffusion-v1-5": For more advanced features (requires more adaptation)
    """
    
    print(f"Creating pretrained diffusion generator with model: {model_name}")
    
    generator = PretrainedDiffusionNoiseGenerator(
        model_name=model_name,
        noise_list=noise_list,
        device=device,
        freeze_backbone=freeze_backbone
    )
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint)
    
    return generator


# Memory-efficient training utilities
class MemoryEfficientTrainer:
    """Utilities for training with limited GPU memory"""
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """Enable gradient checkpointing to trade compute for memory"""
        if hasattr(model.unet, 'enable_gradient_checkpointing'):
            model.unet.enable_gradient_checkpointing()
    
    @staticmethod
    def setup_mixed_precision():
        """Setup mixed precision training"""
        from torch.cuda.amp import GradScaler
        return GradScaler()
    
    @staticmethod
    def get_memory_efficient_optimizer(model, lr=1e-5):
        """Get memory-efficient optimizer settings"""
        # Only train adapter layers and physics parameters if backbone is frozen
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        print(f"Training {len(trainable_params)} parameter groups")
        
        # Use 8-bit optimizer if available
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=0.01)
            print("Using 8-bit AdamW optimizer")
        except:
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
            print("Using standard AdamW optimizer")
        
        return optimizer