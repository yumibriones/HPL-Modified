import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from conch.open_clip_custom import create_model_from_pretrained

def generate_embeddings_from_filenames_npy(filenames_npy_path, output_dir, 
                                           model_path='/gpfs/home/yb2612/.cache/huggingface/hub/models--MahmoodLab--CONCH/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin'):
    """
    Generate CONCH image embeddings using image_filenames.npy as input.
    
    Args:
        filenames_npy_path (str): Path to the .npy file containing image filepaths to embed.
        output_dir (str): Directory to save the output embeddings.
        model_path (str): Path to the pretrained CONCH model checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=model_path)
    model.eval().cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load .npy file
    target_filenames = np.load(filenames_npy_path, allow_pickle=True).tolist()

    all_embeddings = []
    matched_filenames = []

    for path in tqdm(target_filenames, desc=f"Processing {output_dir}"):
        try:
            img = Image.open(path).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.inference_mode():
                embedding = model.encode_image(image_tensor, proj_contrast=True, normalize=True)
                all_embeddings.append(embedding.cpu().numpy())
                matched_filenames.append(path)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        np.save(os.path.join(output_dir, "image_embeddings.npy"), embeddings_array)
        np.save(os.path.join(output_dir, "image_filenames.npy"), np.array(matched_filenames))
        print(f"[DONE] Saved embeddings to {output_dir}")
    else:
        print(f"[FAILED] No embeddings were extracted for {output_dir}")

# Example usage
results_dir = "/gpfs/home/yb2612/dl4med_25/dl_project/results/conch"

generate_embeddings_from_filenames_npy(f"{results_dir}/train/image_filenames.npy", 
                                       f"{results_dir}/train")

generate_embeddings_from_filenames_npy(f"{results_dir}/val/image_filenames.npy", 
                                       f"{results_dir}/val")

generate_embeddings_from_filenames_npy(f"{results_dir}/test/image_filenames.npy", 
                                       f"{results_dir}/test")
