# scripts/train_pretrained_diffusion_noisemodel.py
import sys, os, glob
sys.path.append("../.")
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import argparse, json, torchvision
import scipy.io
import helper.canon_supervised_dataset as dset
import helper.pretrained_diffusion_helper as pdh

def main():
    parser = argparse.ArgumentParser(description='Pretrained diffusion noise model fine-tuning')
    
    # Model selection
    parser.add_argument('--pretrained_model', default='google/ddpm-celebahq-256', 
                        help='Pretrained model from HuggingFace')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze pretrained weights, only train adapters')
    
    # Training parameters  
    parser.add_argument('--noiselist', default='shot_read_uniform_row1_rowt_fixed1', 
                        help='Types of noise to include')
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--dataset', default='color_gray')
    parser.add_argument('--batch_size', default=4, type=int, help='Smaller batch for memory efficiency')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='Lower LR for fine-tuning')
    parser.add_argument('--lambda_phys', default=0.1, type=float)
    parser.add_argument('--num_epochs', default=100, type=int, help='Fewer epochs for fine-tuning')
    parser.add_argument('--num_inference_steps', default=20, type=int, help='Fewer steps for speed')
    
    # Memory optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--use_8bit_optimizer', action='store_true', default=False)
    
    # Paths
    parser.add_argument('--save_path', default='../saved_models/')
    parser.add_argument('--notes', default='pretrained_diffusion')
    
    args = parser.parse_args()
    
    # Create save directory
    folder_name = args.save_path + f'pretrained_diffusion_{args.notes}_{args.pretrained_model.replace("/", "_")}/'
    os.makedirs(folder_name, exist_ok=True)
    
    with open(folder_name + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # Single GPU training for simplicity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_single_gpu(args, folder_name, device)

def get_lightweight_dataset(args):
    """Get dataset with memory-efficient loading"""
    # Smaller crop size and simpler transforms
    composed_transforms = torchvision.transforms.Compose([
        dset.ToTensor2(), 
        dset.RandCrop_gen(shape=(args.crop_size, args.crop_size))
    ])
    
    # Load subset of data for faster experimentation
    if 'color' in args.dataset:
        all_files_mat = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[0:20]  # Reduced dataset
        all_files_mat_test = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[20:25]
    
        dataset_train = dset.Get_sample_batch(all_files_mat, composed_transforms)
        dataset_test = dset.Get_sample_batch(all_files_mat_test, composed_transforms)
    else:
        # Fallback to gray dataset
        filepath_noisy = '../data/paired_data/graybackground_mat/'
        dataset_train = dset.Get_sample_noise_batch(filepath_noisy, composed_transforms, fixed_noise=False)
        dataset_test = dataset_train  # Use same for quick testing
        
    return dataset_train, dataset_test

def train_single_gpu(args, folder_name, device):
    """Memory-efficient single GPU training"""
    print(f'Training on device: {device}')
    
    # Load pretrained model
    print(f'Loading pretrained model: {args.pretrained_model}')
    try:
        generator = pdh.load_pretrained_diffusion_generator(
            model_name=args.pretrained_model,
            noise_list=args.noiselist,
            device=device,
            freeze_backbone=args.freeze_backbone
        )
    except Exception as e:
        print(f"Could not load {args.pretrained_model}, falling back to ddpm-celebahq-256")
        generator = pdh.load_pretrained_diffusion_generator(
            model_name="google/ddpm-celebahq-256",
            noise_list=args.noiselist,
            device=device,
            freeze_backbone=args.freeze_backbone
        )
    
    generator.to(device)
    
    # Memory optimizations
    if args.gradient_checkpointing:
        pdh.MemoryEfficientTrainer.enable_gradient_checkpointing(generator)
    
    # Setup mixed precision
    scaler = None
    if args.mixed_precision:
        scaler = pdh.MemoryEfficientTrainer.setup_mixed_precision()
    
    # Optimizer - only train unfrozen parameters
    optimizer = pdh.MemoryEfficientTrainer.get_memory_efficient_optimizer(generator, args.lr)
    
    # Dataset
    dataset_train, dataset_test = get_lightweight_dataset(args)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced workers
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    print("Starting training...")
    
    total_losses = []
    noise_losses = []
    physics_losses = []
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        generator.train()
        epoch_losses = []
        
        for batch_idx, sample in enumerate(train_loader):
            # Get clean images
            if 'gt_label_nobias' in sample:
                clean_images = sample['gt_label_nobias'].to(device, non_blocking=True)
            else:
                clean_images = sample['gt_label'].to(device, non_blocking=True)
            
            # Handle temporal dimension
            if len(clean_images.shape) == 5:
                B, C, T, H, W = clean_images.shape
                clean_images = clean_images.transpose(1, 2).reshape(B * T, C, H, W)
            
            generator.indices = sample.get('rand_inds', None)
            
            # Forward pass with mixed precision
            if args.mixed_precision and scaler is not None:
                with autocast():
                    total_loss, noise_loss, physics_loss = generator(clean_images)
                    total_loss = total_loss / args.gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(total_loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                total_loss, noise_loss, physics_loss = generator(clean_images)
                total_loss = total_loss / args.gradient_accumulation_steps
                
                total_loss.backward()
                
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_losses.append(total_loss.item() * args.gradient_accumulation_steps)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item() * args.gradient_accumulation_steps:.6f}')
        
        # Record epoch metrics
        avg_loss = np.mean(epoch_losses)
        total_losses.append(avg_loss)
        
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.6f}')
        
        # Evaluation and saving
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            # Quick evaluation
            generator.eval()
            with torch.no_grad():
                eval_losses = []
                for sample in test_loader:
                    if 'gt_label_nobias' in sample:
                        clean_images = sample['gt_label_nobias'].to(device)
                    else:
                        clean_images = sample['gt_label'].to(device)
                    
                    if len(clean_images.shape) == 5:
                        B, C, T, H, W = clean_images.shape
                        clean_images = clean_images.transpose(1, 2).reshape(B * T, C, H, W)
                    
                    total_loss, _, _ = generator(clean_images)
                    eval_losses.append(total_loss.item())
                    break  # Just one batch for quick eval
                
                eval_loss = np.mean(eval_losses)
                print(f'Evaluation loss: {eval_loss:.6f}')
                
                # Generate sample
                generator.eval()
                sample_clean = clean_images[:1]  # Take first sample
                generated_noisy = generator(sample_clean, w=1.0)
                
                # Save sample image
                if isinstance(generated_noisy, torch.Tensor):
                    sample_img = generated_noisy[0].cpu().numpy().transpose(1, 2, 0)[..., :3]
                    sample_img = np.clip(sample_img, 0, 1)
                    Image.fromarray((sample_img * 255).astype(np.uint8)).save(
                        folder_name + f'sample_epoch_{epoch}.jpg'
                    )
            
            # Save checkpoint
            if eval_loss < best_loss:
                best_loss = eval_loss
                checkpoint_path = folder_name + f'best_pretrained_diffusion_epoch_{epoch}.pt'
                torch.save(generator.state_dict(), checkpoint_path)
                print(f'New best model saved: {best_loss:.6f}')
            
            # Regular checkpoint
            checkpoint_path = folder_name + f'checkpoint_epoch_{epoch}.pt'
            torch.save(generator.state_dict(), checkpoint_path)
            
            # Save training progress
            scipy.io.savemat(folder_name + 'training_progress.mat', {
                'total_losses': total_losses,
                'epoch': epoch
            })
    
    print("Training completed!")

if __name__ == '__main__':
    main()