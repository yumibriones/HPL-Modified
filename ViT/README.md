## Vision Transformer (ViT) Training with SSL
Alyssa Pradhan 

### Overview
We attempted to replicate the HPL pipeline using a ViT and trialed several SSL techniques. Initially, we attempted to use the Barlow Twins architecture outlined in the original HPL paper. However, this model did not converge. Subsequently, we implemented self-distillation with no labels (DINO). 

Self attention maps were extracted from the CLS token of the best performing DINO-ViT model to visualise which parts of the histopathology slides the model was focusing on. 

### Repo Structure
`DINO` - code for implementation of DINO-ViT

`barlow_twins` - code for complete and simplified Barlow Twins ViT

`self_attn_maps` - code for generation, and output of visualisation from the self attention maps

## DINO-ViT Architecture Overview

The implementation consists of the following key components:

### 1. Vision Transformer Backbone

The backbone of our model is a Vision Transformer (ViT) that processes histopathology patches:

```
FeatureExtractorVisionTransformer:
- Parameters: image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim
- Default configuration: 224px images, 16px patches, 6 layers, 6 heads, 384 embedding dimension
```

This model divides input images into fixed-size patches, linearly embeds them, adds position embeddings, and processes them through transformer layers to extract features.

### 2. DINO Head

The projection head maps backbone features to the output space:

```
DINOHead:
- MLP with hidden dimensions: 384 → 384 → 256
- Final projection layer: 256 → 65536
- Features normalized before the final projection
```

This head transforms the backbone's output into a high-dimensional space where the self-distillation objective is applied.

### 3. Multi-Crop Wrapper

The `MultiCropWrapper` handles various image crops (global and local) and routes them through the backbone and head:

```
MultiCropWrapper:
- Combines the backbone and head
- Processes multiple image crops of different resolutions
```

### 4. Teacher-Student Architecture

DINO uses two networks:
- **Student network**: Learns from the teacher via self-distillation
- **Teacher network**: Updated via exponential moving average (EMA) of student parameters

## Self-Supervised Training Approach

### Data Augmentation

We use the multi-crop strategy with:
- 2 global crops (224×224 pixels)
- Optional local crops (smaller crops for additional views)

Global crops are processed by both teacher and student networks, while local crops (if used) are only processed by the student.

### DINO Loss Function

The loss function is a cross-entropy between the teacher and student outputs:

```
Loss(student, teacher) = -∑ (softmax(teacher_output/τ_t) * log_softmax(student_output/τ_s))
```

Where:
- τ_t: Teacher temperature (default: 0.04)
- τ_s: Student temperature (default: 0.1)

The center vector prevents representation collapse by centering the teacher outputs.

### Optimization

```
- Optimizer: AdamW
- Learning rate: 0.0005
- Weight decay: 0.04
- Learning rate schedule: Cosine decay with warmup
- EMA update for teacher: momentum = 0.99
```

## References

- DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
