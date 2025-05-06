print("Setting up...")
import numpy as np
import pandas as pd
import ast
import umap
import math
import os
import scanpy as sc
import anndata as ad

# load data
def load_data(metadata_file, embedding_file, filenames_file):
    """
    Load metadata, embeddings, and filenames and merge them based on filepaths.
    
    Parameters:
    - metadata_file: Path to the CSV file containing metadata.
    - embedding_file: Path to the .npy file containing the embeddings.
    - filenames_file: Path to the .npy file containing the corresponding filenames.
    
    Returns:
    - Merged DataFrame with embeddings and metadata.
    """
    print("Loading metadata and embeddings...")
    # load metadata
    metadata = pd.read_csv(metadata_file)
    metadata['filepath'] = metadata.apply(
        lambda row: f"/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/{row['original_set']}/{row['slides']}/{row['tiles']}",
        axis=1
    )
    
    # replace 'valid' with 'val' in filepath
    metadata['filepath'] = metadata['filepath'].str.replace('valid', 'val')

    # load embeddings and filenames
    embeddings = np.load(embedding_file, allow_pickle=True)
    filepaths = np.load(filenames_file, allow_pickle=True)

    # convert embeddings to df
    img_z_latent = [emb for emb in embeddings]
    embedding_df = pd.DataFrame({
        "filepath": filepaths,
        "img_z_latent": img_z_latent
    })
    
    # merge embeddings with metadata
    merged_df = metadata.merge(embedding_df, on="filepath", how="inner")
    return merged_df

def run_umap(merged_df, n_neighbors=30, min_dist=0.0, n_components=2, random_state=42):
    """
    Perform UMAP transformation on the img_z_latent column of the merged dataframe.
    
    Parameters:
    - merged_df: DataFrame with the img_z_latent column.
    - n_neighbors, min_dist, n_components, random_state: UMAP hyperparameters.
    
    Returns:
    - DataFrame with UMAP results added.
    """
    # clean 'img_z_latent' column
    merged_df['img_z_latent'] = merged_df['img_z_latent'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    img_z_latent = pd.DataFrame(merged_df['img_z_latent'].to_list())

    # perform UMAP
    print("Running UMAP...")
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, low_memory=True)
    umap_result = umap_model.fit_transform(img_z_latent)

    # add UMAP results to df
    merged_df['umap_1'] = umap_result[:, 0]
    merged_df['umap_2'] = umap_result[:, 1]
    
    return merged_df
    
def run_leiden(merged_df, resolution=2.0):
    """
    Run Leiden clustering using Scanpy on img_z_latent and append the labels to the dataframe.
    
    Parameters:
    - merged_df: DataFrame with the img_z_latent column.
    - resolution: Resolution parameter for the Leiden algorithm.
    
    Returns:
    - DataFrame with a new column 'leiden_{resolution}' for clustering labels.
    """
    print("Running Leiden clustering...")
    
    # prepare latent embedding matrix
    merged_df['img_z_latent'] = merged_df['img_z_latent'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    X = np.vstack(merged_df['img_z_latent'].to_numpy())

    # create anndata object
    adata = ad.AnnData(X)

    # build neighborhood graph
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=250, method='umap')

    # leiden clustering
    sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}')

    # add clusters to df
    merged_df[f'leiden_{resolution}'] = adata.obs[f'leiden_{resolution}'].values

    return merged_df


# execute
metadata_file = "/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/lung_subsample_clinical_clusters.csv"

for model in ["hpl", "conch", "hpl-clip-scratch"]:
    for set in ["test", "val", "train"]:
        print(f"Processing {model}/{set} data...")
        embedding_file = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}/image_embeddings.npy"
        filenames_file = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}/image_filenames.npy"
        save_dir = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}"

        merged_df = load_data(metadata_file, embedding_file, filenames_file)
        merged_df = run_umap(merged_df)
        merged_df = run_leiden(merged_df)

        # save df
        output_file = os.path.join(save_dir, "umap_leiden_results.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Saved UMAP + Leiden results to {output_file}")