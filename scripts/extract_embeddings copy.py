import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import open_clip

lung_subsample = pd.read_csv("/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/LUAD vs LUSC lung type/TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_2p0__fold4_subsample.csv", header=0)
print(lung_subsample.head())

lung_subsample = lung_subsample[['original_set', 'slides', 'tiles']].copy()

lung_subsample['filepath'] = lung_subsample.apply(
    lambda row: f"/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/{row['original_set']}/{row['slides']}/{row['tiles']}",
    axis=1
)

# Replace 'valid' with 'val' in the 'filepath' column
lung_subsample['filepath'] = lung_subsample['filepath'].str.replace('valid', 'val')

lung_train = lung_subsample[lung_subsample['original_set'] == 'train'].reset_index(drop=True)
print("train:", lung_train.shape)
lung_val = lung_subsample[lung_subsample['original_set'] == 'valid'].reset_index(drop=True)
print("val:", lung_val.shape)
lung_test = lung_subsample[lung_subsample['original_set'] == 'test'].reset_index(drop=True)
print("test:", lung_test.shape)

# 1. Create the model (no pretrained weights)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 2. Load the checkpoint
checkpoint_path = '/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/logs/HPL-ViT-B-32_train_250k/checkpoints/epoch_30.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)
model.eval().cuda()

#### TEST ####

# # Load CSV with full filepaths
# df = lung_test  # change to your actual file
# image_paths = df["filepath"].tolist()  # or whatever the column name is

# all_embeddings = []
# image_names = []

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# model.eval()

# for path in tqdm(image_paths):
#     try:
#         img = Image.open(path).convert("RGB")
#         image_tensor = preprocess(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             embedding = model.encode_image(image_tensor)
#             embedding = embedding / embedding.norm(dim=-1, keepdim=True)
#             all_embeddings.append(embedding.cpu().numpy())
#             image_names.append(path)

#     except Exception as e:
#         print(f"[ERROR] {path}: {e}")

# # Save embeddings and filepaths
# if all_embeddings:
#     embeddings_array = np.concatenate(all_embeddings, axis=0)
#     np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_test_subsample_clip_embeddings.npy", embeddings_array)
#     np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_test_subsample_filenames.npy", np.array(image_names))
# else:
#     print("No embeddings were extracted. Check paths or model/device.")

#### VAL ####

# Load CSV with full filepaths
df = lung_val  # change to your actual file
image_paths = df["filepath"].tolist()  # or whatever the column name is

all_embeddings = []
image_names = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

for path in tqdm(image_paths):
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

# Save embeddings and filepaths
if all_embeddings:
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_val_subsample_clip_embeddings.npy", embeddings_array)
    np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_val_subsample_filenames.npy", np.array(image_names))
else:
    print("No embeddings were extracted. Check paths or model/device.")


#### TRAIN ####

# Load CSV with full filepaths
df = lung_train  # change to your actual file
image_paths = df["filepath"].tolist()  # or whatever the column name is

all_embeddings = []
image_names = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

for path in tqdm(image_paths):
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

# Save embeddings and filepaths
if all_embeddings:
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_train_subsample_clip_embeddings.npy", embeddings_array)
    np.save("/gpfs/home/yb2612/dl4med_25/dl_project/results/lung_train_subsample_filenames.npy", np.array(image_names))
else:
    print("No embeddings were extracted. Check paths or model/device.")