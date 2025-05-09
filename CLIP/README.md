# HPL-CLIP

![image](HPL-CLIP_diagram.png)
*Image adapted from [Quiros et al.](https://www.nature.com/articles/s41467-024-48666-7) and [Radford et al.](https://arxiv.org/pdf/2103.00020)*

How to run HPL-CLIP. This is an integration of [open_clip](https://github.com/mlfoundations/open_clip) with the [Histomorphological-Phenotype-Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning) pipeline.

## Prepare inputs

1. Download train, validation, and test sets from: [LUAD & LUSC 250K subsample](https://drive.google.com/drive/folders/1FuPkMnv6CiDe26doUXfEfQEWShgbmp9P) and [LUAD & LUSC datasets](https://drive.google.com/drive/folders/18skVh8Vk6zoxG3Se5Vlb7a3EKP2xHXXd).

2. Generate image captions using [generate_short_varied_captions.ipynb](https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/notebooks/generate_short_varied_captions.ipynb).

3. Extract all images from HDF5 files using [extract_hdf5_images.ipynb](https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/notebooks/extract_hdf5_images.ipynb).

## Train/validate HPL-CLIP

1. Clone the open_clip repo (https://github.com/mlfoundations/open_clip) to your directory of choice.

2. From the HPL-Modified repo, open [`scripts/bash/HPL-RN50_train_250k_val.sh`](https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/scripts/bash/HPL-RN50_train_250k_val.sh). Edit as needed (e.g., activate correct conda environment, specify correct paths to `open_clip` repo, `logs`, `train-data`, `val-data`, etc.).

3. Submit [`scripts/bash/HPL-RN50_train_250k_val.sh`](https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/scripts/bash/HPL-RN50_train_250k_val.sh) as a batch job. A TensorBoard log should appear in specified log directory (e.g., `/gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/logs/`).

4. Check for the TensorBoard log in the log directory (e.g.,`events.out.tfevents.1745262456.gpu-0003.121832.0`).

5. Forward port (e.g., 9199) from your remote machine to your local machine.

```
ssh -L 9199:localhost:9199 bigpurple
```

6. Activate a conda environment which has TensorBoard installed (e.g., `dl4med_25`).

```
conda activate dl4med_25
```

7. Run TensorBoard and point it to the log directory that contains the event files:

```
tensorboard --logdir /gpfs/home/yb2612/dl4med_25/dl_project/data/scratch_data/hpl-clip/logs/ --port 9199 --host 0.0.0.0
```

This should return something like:

```
TensorFlow installation not found - running with reduced feature set.
TensorBoard 2.19.0 at http://0.0.0.0:9199/ (Press CTRL+C to quit)
```

8. Open the link from the output above in a web browser (e.g., http://0.0.0.0:9199/). This opens the TensorBoard interface, which contains training metrics, graphs, etc. Use these graphs to select a final HPL-CLIP model.

## Test HPL-CLIP

1. Generate embeddings from the selected HPL-CLIP model using [extract_embeddings_clip.py](https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/scripts/py/extract_embeddings_clip.py).*
2. Run UMAP/Leiden clustering on embeddings using [run_umap_leiden.py](https://github.com/yumibriones/HPL-Modified/blob/main/evaluation/run_umap_leiden.py).*
3. Plot UMAP with clustering results/clinical features overlaid on top using [plot_umap.py](https://github.com/yumibriones/HPL-Modified/blob/main/evaluation/plot_umap.py).*

*If submitting as a batch job on HPC, use corresponding scripts in [scripts/bash](https://github.com/yumibriones/HPL-Modified/tree/main/CLIP/scripts/bash).

