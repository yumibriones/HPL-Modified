import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from conch.open_clip_custom import create_model_from_pretrained

def generate_conch_embeddings(csv_path, output_dir, 
                              model_path='/gpfs/home/yb2612/.cache/huggingface/hub/models--MahmoodLab--CONCH/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin'):
    """
    Generate CONCH image embeddings from a CSV and save to output_dir.
    
    Args:
        csv_path (str): Path to CSV file containing a column 'filepath' with image paths.
        output_dir (str): Directory to save the output embeddings and filenames.
        model_path (str): Path to the CONCH pretrained model checkpoint.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=model_path)
    model.eval().cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load CSV
    df = pd.read_csv(csv_path)
    image_paths = df["filepath"].tolist()

    all_embeddings = []
    image_names = []

    for path in tqdm(image_paths, desc=f"Processing {output_dir}"):
        try:
            img = Image.open(path).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.inference_mode():
                embedding = model.encode_image(image_tensor, proj_contrast=True, normalize=True)
                all_embeddings.append(embedding.cpu().numpy())
                image_names.append(path)

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    # Save embeddings
    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        np.save(os.path.join(output_dir, "image_embeddings.npy"), embeddings_array)
        np.save(os.path.join(output_dir, "image_filenames.npy"), np.array(image_names))
        print(f"[DONE] Saved embeddings to {output_dir}")
    else:
        print(f"[FAILED] No embeddings were extracted from {csv_path}")

caption_dir = "/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/long_consistent_captions"
results_dir = "/gpfs/home/yb2612/dl4med_25/dl_project/results/conch"
# generate_conch_embeddings(f"{caption_dir}/lung_train_subsample_filepath_caption.csv", f"{results_dir}/train")
generate_conch_embeddings(f"{caption_dir}/lung_val_subsample_filepath_caption.csv", f"{results_dir}/val")
generate_conch_embeddings(f"{caption_dir}/lung_test_subsample_filepath_caption.csv", f"{results_dir}/test")