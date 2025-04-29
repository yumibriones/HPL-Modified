import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import open_clip

def generate_clip_embeddings(filelist_path, model, preprocess, output_path):
    """
    Generate CLIP image embeddings and save results.

    Args:
        filelist_path (str): Path to .npy file containing image paths.
        model: Loaded CLIP model.
        preprocess: CLIP image preprocessing function.
        output_path (str): Directory to save embeddings and filenames.
    """
    os.makedirs(output_path, exist_ok=True)

    # Load filenames to include
    filenames = np.load(filelist_path, allow_pickle=True).tolist()

    # Setup
    all_embeddings = []
    image_names = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for path in tqdm(filenames, desc=f"Encoding {os.path.basename(output_path)}"):
        try:
            img = Image.open(path).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                all_embeddings.append(embedding.cpu().numpy())
                image_names.append(path)

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        np.save(os.path.join(output_path, "image_embeddings.npy"), embeddings_array)
        np.save(os.path.join(output_path, "image_filenames.npy"), np.array(image_names))
        print(f"[DONE] Saved {len(all_embeddings)} embeddings to {output_path}")
    else:
        print(f"[FAILED] No embeddings extracted for {filelist_path}")


### Setup model ###
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained=None)
tokenizer = open_clip.get_tokenizer('RN50')
checkpoint_path = '/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/logs/HPL-RN50_train_250k_val_varied/checkpoints/epoch_16.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)

### Paths ###
results_dir = "/gpfs/home/yb2612/dl4med_25/dl_project/results/clip-scratch"

### Generate embeddings for each split ###
generate_clip_embeddings(
    filelist_path=f"{results_dir}/train/image_filenames.npy",
    model=model,
    preprocess=preprocess,
    output_path=f"{results_dir}/train"
)

generate_clip_embeddings(
    filelist_path=f"{results_dir}/val/image_filenames.npy",
    model=model,
    preprocess=preprocess,
    output_path=f"{results_dir}/val"
)

generate_clip_embeddings(
    filelist_path=f"{results_dir}/test/image_filenames.npy",
    model=model,
    preprocess=preprocess,
    output_path=f"{results_dir}/test"
)
