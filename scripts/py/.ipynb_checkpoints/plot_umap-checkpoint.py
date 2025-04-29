print("Setting up...")
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os

# dimplot for leiden clusters
def dim_plot(merged_df, umap_1_col='umap_1', umap_2_col='umap_2', feature_col='leiden_2.0', save_dir='.'):
    """
    Plot UMAP with specified feature column and save the plot.
    
    Parameters:
    - merged_df: DataFrame containing the UMAP results and feature column.
    - umap_1_col, umap_2_col: Columns with UMAP coordinates.
    - feature_col: Column to color the UMAP points by.
    """
    print(f"Plotting UMAP by {feature_col}...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_1_col, y=umap_2_col, hue=feature_col, data=merged_df, palette='tab20', legend=None, s=5, alpha=0.7)
    plt.title(f'UMAP Colored by {feature_col}')
    plt.savefig(f"{save_dir}/umap_by_{feature_col}.png", bbox_inches='tight')
    plt.close()

# feature plot to highlight one feature at a time
def feature_plot(merged_df, umap_1_col='umap_1', umap_2_col='umap_2', feature_cols='leiden_2.0', save_dir='.'):
    print(f"Plotting UMAP by {feature_cols}...")
    # Loop over each feature column and plot
    for feature in feature_cols:
        unique_values = merged_df[feature].dropna().value_counts().index.tolist()
        n_vals = len(unique_values)
        n_cols = 3
        n_rows = math.ceil(n_vals / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, val in enumerate(unique_values):
            ax = axes[i]
            base = merged_df.copy()
            base['plot_group'] = 'Other'
            base.loc[base[feature] == val, 'plot_group'] = val
            sns.scatterplot(data=base, x=umap_1_col, y=umap_2_col, hue='plot_group', palette={val: 'red', 'Other': 'lightgray'}, ax=ax, s=5, alpha=0.7, legend=False)
            ax.set_title(f'{val}')
            ax.set_xlabel('')
            ax.set_ylabel('')
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f'UMAP by {feature}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/umap_by_{feature}.png", bbox_inches='tight')
        plt.close()

# plot by age
def plot_umap_by_age(merged_df, umap_1_col='umap_1', umap_2_col='umap_2', age_col='age_at_initial_pathologic_diagnosis', save_dir='.'):
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = Normalize(vmin=merged_df[age_col].min(), vmax=merged_df[age_col].max())
    cmap = cm.plasma
    colors = cmap(norm(merged_df[age_col]))
    sc = ax.scatter(merged_df[umap_1_col], merged_df[umap_2_col], c=colors, s=3, alpha=0.5)
    ax.set_title(f'UMAP Colored by {age_col}')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=age_col)
    plt.savefig(f"{save_dir}/umap_by_{age_col}.png", bbox_inches='tight')
    plt.close()

# execute
metadata_file = "/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/lung_subsample_clinical_clusters.csv"

for model in ["hpl", "conch", "hpl-clip-scratch"]:
    for set in ["test", "val", "train"]:
        print(f"Processing {model}/{set} data...")
        embedding_file = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}/image_embeddings.npy"
        filenames_file = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}/image_filenames.npy"
        save_dir = f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}"
        merged_df = pd.read_csv(f"/gpfs/home/yb2612/dl4med_25/dl_project/results/{model}/{set}/umap_leiden_results.csv")
        dim_plot(merged_df, feature_col='leiden_2.0', save_dir=save_dir)
        dim_plot(merged_df, feature_col='sampleID', save_dir=save_dir)
        feature_plot(merged_df, feature_cols=['_primary_disease', 'gender', 'tobacco_smoking_history_label'], save_dir=save_dir)
        plot_umap_by_age(merged_df, age_col='age_at_initial_pathologic_diagnosis', save_dir=save_dir)