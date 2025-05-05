import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
import os

class DINOAttentionVisualizer:
    def __init__(self, model):
        """
        Initialize the visualizer with the model
        
        Args:
            model: The loaded and initialized model
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        self.attention_maps = {i: [] for i in range(6)}  # One for each layer
        self.hooks = []
        
        # Register hooks for each encoder layer's self-attention module
        for i in range(6):  # 6 layers as per your --depth parameter
            layer_name = f'encoder_layer_{i}'
            # Adjust path based on your model structure - print debug info
            print(f"Looking for layer: {layer_name}")
            layer = getattr(self.model.backbone.vit.encoder.layers, layer_name)
            print(f"Found layer: {layer}")
            
            # Register the hook to capture attention
            print(f"Registering hook for layer {i}")
            hook = layer.self_attention.register_forward_hook(
                self.get_attention_hook(i)
            )
            self.hooks.append(hook)
            print(f"Hook registered successfully for layer {i}")
    
    def get_attention_hook(self, layer_idx):
        def hook(module, input, output):
            print(f"Hook called for layer {layer_idx}")
            # For DINO ViT, we need to extract the attention matrix
            # The implementation below works for both the standard ViT and DINO's implementation
            
            # Get input to the self-attention module
            x = input[0]  # Shape: [batch_size, num_patches + 1, embed_dim]
            print(f"Input shape: {x.shape}")
            
            # Compute attention explicitly
            # This works for the standard self-attention implementation
            # Get the query, key projections
            if hasattr(module, 'in_proj_weight'):
                # Combined QKV projection
                qkv_weight = module.in_proj_weight
                qkv_bias = module.in_proj_bias
                
                # Split into Q, K, V
                dim = qkv_weight.size(0) // 3
                q_weight, k_weight, v_weight = torch.split(qkv_weight, dim, dim=0)
                q_bias, k_bias, v_bias = torch.split(qkv_bias, dim, dim=0) if qkv_bias is not None else (None, None, None)
                
                # Project Q, K
                q = F.linear(x, q_weight, q_bias)
                k = F.linear(x, k_weight, k_bias)
                
                # Compute attention
                q = q.reshape(x.shape[0], x.shape[1], module.num_heads, -1).transpose(1, 2)
                k = k.reshape(x.shape[0], x.shape[1], module.num_heads, -1).transpose(1, 2)
                
                # Compute attention scores
                attn = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
                attn = F.softmax(attn, dim=-1)
                
            else:
                print("Module structure doesn't match expected pattern")
                print(f"Module attributes: {dir(module)}")
                # Try different approach if the first one doesn't work
                # ...
            
            # Store attention map
            print(f"Attention map shape: {attn.shape}")
            self.attention_maps[layer_idx].append(attn.detach())
        
        return hook
    
    def cleanup(self):
        # Remove registered hooks
        for hook in self.hooks:
            hook.remove()
        
        # Clear stored attention maps
        for layer_idx in self.attention_maps:
            self.attention_maps[layer_idx] = []
    
    def extract_attention(self, img_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Clear previous attention maps
        for layer_idx in self.attention_maps:
            self.attention_maps[layer_idx] = []
        
        # Load and preprocess the image
        img = Image.open(img_path).convert('RGB')
        print(f"Loaded image: {img_path}, size: {img.size}")
        
        # Apply transformations similar to those used during training
        transform = transforms.Compose([
            transforms.Resize(224),  # As per your --patch_size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        print(f"Image tensor shape: {img_tensor.shape}")
        
        # Pass image through the model
        print("Running forward pass")
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        print(f"Forward pass complete. Collected attention maps for {len(self.attention_maps)} layers.")
        for layer_idx, maps in self.attention_maps.items():
            print(f"Layer {layer_idx}: {len(maps)} attention maps")
            if maps:
                print(f"  Shape: {maps[0].shape}")
        
        # Return the original image and collected attention maps
        return img, self.attention_maps
    
    def visualize_attention(self, img_path, layer_idx=5, head_idx=0, save_path=None, 
                           device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Visualize attention map from a specific layer and head
        
        Args:
            img_path: Path to the input image
            layer_idx: Index of the transformer layer (0-5)
            head_idx: Index of the attention head (0-7)
            save_path: Path to save the visualization (optional)
            device: Device to run the model on
        """
        print(f"Visualizing attention for layer {layer_idx}, head {head_idx}")
        
        # Create output directory if it doesn't exist
        if save_path is None:
            output_dir = 'attention_visualizations'
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'attention_layer{layer_idx}_head{head_idx}.png')
        
        # Extract attention maps
        img, attention_maps = self.extract_attention(img_path, device)
        
        # Get the attention map for the specified layer
        if not attention_maps[layer_idx]:
            print(f"No attention maps found for layer {layer_idx}")
            return None, None
        
        # Get attention map from the specified head
        attn = attention_maps[layer_idx][0]  # First batch item
        print(f"Attention tensor shape: {attn.shape}")
        
        # Handle different attention map shapes
        if len(attn.shape) == 4:  # [batch, heads, seq, seq]
            print(f"Multi-head attention with {attn.shape[1]} heads")
            attn = attn[0, head_idx]  # Get specified head
        else:  # [batch, seq, seq]
            print("Single-head attention")
            attn = attn[0]  # Just get first batch
        
        print(f"Selected attention shape: {attn.shape}")
        
        # For ViT, we're interested in the attention from CLS token to patch tokens
        # CLS token is at index 0
        attn_from_cls = attn[0, 1:]  # Shape: [num_patches]
        print(f"CLS token attention shape: {attn_from_cls.shape}")
        
        # Reshape to match the image patches grid
        # For patch size 16 in a 224x224 image, we get 14x14 patches
        patches_per_side = 224 // 16  # = 14 for your config
        print(f"Reshaping to {patches_per_side}x{patches_per_side} grid")
        attn_map = attn_from_cls.reshape(patches_per_side, patches_per_side).cpu()
        
        # Upscale attention map to original image size
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=img.size[::-1],  # (height, width)
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize attention map
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Create heatmap
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(img))
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention map
        plt.subplot(1, 3, 2)
        plt.imshow(attn_map, cmap='inferno')
        plt.title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        img_array = np.array(img) / 255.0
        attn_map_rgb = plt.cm.inferno(attn_map)[:, :, :3]
        overlay = 0.6 * attn_map_rgb + 0.4 * img_array
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        # Don't display in non-interactive environment
        plt.close()
        
        return attn_map, overlay



# Example usage:
import torch
from train_dino2 import FeatureExtractorVisionTransformer, MultiCropWrapper, DINOHead  

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create just the teacher model
teacher_backbone = FeatureExtractorVisionTransformer(
        image_size=224, 
        patch_size=16,
        num_layers=6,
        num_heads=8,
        hidden_dim=512,
        mlp_dim=512 * 4
    )

teacher_head = DINOHead(
        in_dim=512,
        out_dim=32768,
        hidden_dim=512,
        bottleneck_dim=256,
        use_bn=True
    )
teacher = MultiCropWrapper(teacher_backbone, teacher_head)

# Load the checkpoint
print("Loading model checkpoint...")
model_path = "/gpfs/home/amp10295/scratch_dl/ssl_vit/dino_v2/dino_model_best.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
print("Checkpoint loaded successfully")
print(f"Checkpoint keys: {checkpoint.keys()}")

print("Creating teacher model...")

teacher.load_state_dict(checkpoint['teacher_state_dict'])
teacher.eval()  # Set to evaluation mode
print("Teacher model created")

print("Creating visualizer...")
visualizer = DINOAttentionVisualizer(teacher)
print("Visualizer created successfully")

# Process image and generate visualizations for multiple layers and heads
image_path = "/gpfs/home/amp10295/final_dl/test/TCGA-05-4382-01Z-00-DX1/2_7.jpeg"
print(f"Processing image: {image_path}")

# Create an output directory for all visualizations
output_dir = 'attention_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Visualize last layer attention for multiple heads
for head_idx in range(8):  # 8 heads as per your --num_heads parameter
    save_path = os.path.join(output_dir, f'attention_layer5_head{head_idx}.png')
    print(f"Visualizing Layer 5, Head {head_idx}")
    visualizer.visualize_attention(
        image_path, 
        layer_idx=5, 
        head_idx=head_idx,
        save_path=save_path
    )

# Visualize attention at different layers using head 0
for layer_idx in range(6):  # 6 layers as per your --depth parameter
    save_path = os.path.join(output_dir, f'attention_layer{layer_idx}_head0.png')
    print(f"Visualizing Layer {layer_idx}, Head 0")
    visualizer.visualize_attention(
        image_path, 
        layer_idx=layer_idx, 
        head_idx=0,
        save_path=save_path
    )

# Clean up
print("Cleaning up...")
visualizer.cleanup()
print("Finished successfully")
