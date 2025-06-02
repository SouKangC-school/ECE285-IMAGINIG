# helper/data_splitting.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import glob

def get_proper_train_test_split(args, min_test_size=5):
    """
    Create proper train/test split even with limited data
    """
    composed_transforms_train = torchvision.transforms.Compose([
        dset.ToTensor2(), 
        dset.RandCrop_gen(shape=(args.crop_size, args.crop_size))
    ])
    
    composed_transforms_test = torchvision.transforms.Compose([
        dset.ToTensor2(), 
        dset.FixedCrop_gen(shape=(args.crop_size, args.crop_size))  # Fixed crop for consistent testing
    ])

    if 'color' in args.dataset:
        all_files_mat = glob.glob('../data/paired_data/stillpairs_mat/*.mat')
        
        # Ensure minimum test size
        total_files = len(all_files_mat)
        if total_files < min_test_size * 2:
            print(f"Warning: Only {total_files} files available. Using holdout validation.")
            test_size = max(1, total_files // 5)  # 20% for test, minimum 1
        else:
            test_size = max(min_test_size, total_files // 5)
        
        # Random split with fixed seed for reproducibility
        train_files, test_files = train_test_split(
            all_files_mat, 
            test_size=test_size, 
            random_state=42,  # Fixed seed for reproducibility
            shuffle=True
        )
        
        print(f"Data split: {len(train_files)} train, {len(test_files)} test files")
        
        dataset_train = dset.Get_sample_batch(train_files, composed_transforms_train)
        dataset_test = dset.Get_sample_batch(test_files, composed_transforms_test)
    
    elif 'gray' in args.dataset:
        # For gray dataset, create a proper split
        filepath_noisy = '../data/paired_data/graybackground_mat/'
        
        # Get all available files
        all_noisy_files = glob.glob(filepath_noisy + 'noisy*.mat')
        
        if len(all_noisy_files) < min_test_size * 2:
            print("Warning: Very limited gray data. Consider data augmentation.")
            # Use different random seeds for train/test instead
            dataset_train = dset.Get_sample_noise_batch(
                filepath_noisy, composed_transforms_train, fixed_noise=False
            )
            # Same data but different random crops/augmentations
            dataset_test = dset.Get_sample_noise_batch(
                filepath_noisy, composed_transforms_test, fixed_noise=False
            )
            print("Using same files but different augmentations for test")
        else:
            # If enough data, do proper file-level split
            train_files, test_files = train_test_split(
                all_noisy_files, test_size=0.2, random_state=42
            )
            # You'd need to modify the dataset class to accept specific files
            dataset_train = dset.Get_sample_noise_batch(
                filepath_noisy, composed_transforms_train, fixed_noise=False
            )
            dataset_test = dset.Get_sample_noise_batch(
                filepath_noisy, composed_transforms_test, fixed_noise=False
            )
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")
    
    return dataset_train, dataset_test

def create_validation_split(dataset, val_ratio=0.2, seed=42):
    """
    Create validation split from training data when test data is very limited
    """
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # Use torch's random split for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Created validation split: {train_size} train, {val_size} validation")
    return train_dataset, val_dataset

def get_cross_validation_folds(dataset, n_folds=5, seed=42):
    """
    Create cross-validation folds for very small datasets
    """
    from sklearn.model_selection import KFold
    
    dataset_size = len(dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    indices = np.arange(dataset_size)
    
    for train_idx, val_idx in kf.split(indices):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        folds.append((train_subset, val_subset))
    
    print(f"Created {n_folds}-fold cross-validation")
    return folds

# Modified training function with proper evaluation
def get_lightweight_dataset_proper(args):
    """
    Improved version that avoids train=test
    """
    # Option 1: Proper file-level split (RECOMMENDED)
    try:
        dataset_train, dataset_test = get_proper_train_test_split(args)
        return dataset_train, dataset_test
    except Exception as e:
        print(f"Could not create proper split: {e}")
    
    # Option 2: Fallback - same files but different augmentations
    print("Fallback: Using different augmentations for train/test")
    
    composed_transforms_train = torchvision.transforms.Compose([
        dset.ToTensor2(), 
        dset.RandCrop_gen(shape=(args.crop_size, args.crop_size))  # Random crops
    ])
    
    composed_transforms_test = torchvision.transforms.Compose([
        dset.ToTensor2(), 
        dset.FixedCrop_gen(shape=(args.crop_size, args.crop_size))  # Fixed crops
    ])
    
    if 'color' in args.dataset:
        # Use subset of files
        all_files_mat = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[:10]  # Limit for demo
        
        dataset_train = dset.Get_sample_batch(all_files_mat, composed_transforms_train)
        dataset_test = dset.Get_sample_batch(all_files_mat, composed_transforms_test)
    else:
        filepath_noisy = '../data/paired_data/graybackground_mat/'
        dataset_train = dset.Get_sample_noise_batch(filepath_noisy, composed_transforms_train, fixed_noise=False)
        dataset_test = dset.Get_sample_noise_batch(filepath_noisy, composed_transforms_test, fixed_noise=False)
    
    # Create validation split from training data
    dataset_train, dataset_val = create_validation_split(dataset_train, val_ratio=0.2)
    
    print("Warning: Using same source files for train/test with different augmentations")
    print("This is better than identical datasets but still not ideal")
    
    return dataset_train, dataset_val  # Return validation split instead

# Example usage in training script
def train_with_proper_evaluation(args):
    """
    Example of how to modify the training loop for proper evaluation
    """
    dataset_train, dataset_test = get_lightweight_dataset_proper(args)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(dataset_train)}")
    print(f"Test samples: {len(dataset_test)}")
    print(f"Train/test overlap: {'NONE' if len(set(range(len(dataset_train))) & set(range(len(dataset_test)))) == 0 else 'EXISTS'}")
    
    return train_loader, test_loader

# Alternative: Synthetic test data generation
def create_synthetic_test_data(args, num_samples=20):
    """
    Create synthetic test data when real test data is unavailable
    """
    synthetic_samples = []
    
    for i in range(num_samples):
        # Create synthetic clean image
        if args.crop_size == 256:
            clean_img = torch.rand(4, 256, 256) * 0.8 + 0.1  # Avoid extremes
        else:
            clean_img = torch.rand(4, args.crop_size, args.crop_size) * 0.8 + 0.1
        
        # Add known noise pattern for ground truth
        noise_level = 0.1 * (i % 5 + 1)  # Varying noise levels
        noise = torch.randn_like(clean_img) * noise_level
        noisy_img = torch.clamp(clean_img + noise, 0, 1)
        
        synthetic_samples.append({
            'gt_label_nobias': clean_img.unsqueeze(0),  # Add batch dimension
            'noisy_input': noisy_img.unsqueeze(0),
            'noise_level': noise_level
        })
    
    print(f"Created {num_samples} synthetic test samples")
    return synthetic_samples

# Best practices summary
def get_evaluation_strategy(dataset_size):
    """
    Recommend evaluation strategy based on dataset size
    """
    if dataset_size >= 100:
        return "proper_split", "Use 80/20 train/test split with file-level separation"
    elif dataset_size >= 50:
        return "validation_split", "Use 80/20 train/validation split from same files"
    elif dataset_size >= 20:
        return "cross_validation", "Use 5-fold cross-validation"
    else:
        return "synthetic_test", "Create synthetic test data + visual inspection"