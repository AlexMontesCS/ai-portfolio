import torch
import numpy as np
import random
import time
from datetime import datetime
import os

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds):
    """
    Format time in seconds to readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def get_device():
    """
    Get the best available device for training.
    
    Returns:
        torch.device: Device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def create_model_summary(model, input_shape=None):
    """
    Create a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (optional)
        
    Returns:
        dict: Model summary information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model info
    model_info = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'memory_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    # Add layer information
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_params = sum(p.numel() for p in module.parameters())
            layers.append({
                'name': name,
                'type': module.__class__.__name__,
                'parameters': layer_params
            })
    
    model_info['layers'] = layers
    
    return model_info

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        
    Returns:
        dict: Checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def calculate_model_flops(model, input_shape):
    """
    Estimate FLOPs (Floating Point Operations) for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        int: Estimated FLOPs
    """
    # This is a simplified estimation
    # For more accurate FLOP counting, consider using libraries like thop or fvcore
    
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, torch.nn.Conv2d):
            # Convolution FLOPs
            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
            
            active_elements_count = batch_size * int(np.prod(output_dims))
            overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
            
            total_flops += overall_conv_flops
            
        elif isinstance(module, torch.nn.Linear):
            # Linear layer FLOPs
            total_flops += input[0].numel() * module.out_features
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Forward pass with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops

def print_training_info(model, train_loader, test_loader, device):
    """
    Print comprehensive training information.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device being used
    """
    print("=" * 50)
    print("TRAINING INFORMATION")
    print("=" * 50)
    
    # Model info
    model_summary = create_model_summary(model)
    print(f"Model: {model_summary['model_name']}")
    print(f"Total Parameters: {model_summary['total_parameters']:,}")
    print(f"Trainable Parameters: {model_summary['trainable_parameters']:,}")
    print(f"Model Size: {model_summary['memory_size_mb']:.2f} MB")
    
    # Dataset info
    print(f"\nDataset Information:")
    print(f"Training Samples: {len(train_loader.dataset):,}")
    print(f"Test Samples: {len(test_loader.dataset):,}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Training Batches: {len(train_loader)}")
    print(f"Test Batches: {len(test_loader)}")
    
    # Device info
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("=" * 50)

def create_experiment_dir(base_dir="experiments"):
    """
    Create a directory for experiment results.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        str: Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir
