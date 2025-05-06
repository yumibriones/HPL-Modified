# HPL-Modified

Modifications to the [Histomorphological Phenotype Learning pipeline](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning). This pipeline generates histomorphological phenotype clusters (HPCs) from tiled H&E images via unsupervised learning.

Original HPL paper by Quiros et al. is here: https://www.nature.com/articles/s41467-024-48666-7.

## Authors
Yumi Briones - yb2612@nyu.edu, Yumi.Briones@nyulangone.org  
Jennifer Motter - mottej02@nyu.edu, Jennifer.Motter@nyulangone.org  
Alyssa Pradhan - amp10295@nyu.edu, Alyssa.Pradhan@nyulangone.org  

## Repo structure
* `VICReg` - README documentation and files to perform HPL-VICReg and HPL-BarlowTwins replication
* `ViT` - README documentation and files to perform HPL-ViT
* `CLIP` - README documentation and files to perform HPL-CLIP and HPL-CONCH

## Data

All data are from https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning.

1. For initial training, we used a 250k subsample of LUAD and LUSC samples: [LUAD & LUSC 250K subsample](https://drive.google.com/drive/folders/1FuPkMnv6CiDe26doUXfEfQEWShgbmp9P)
2. For complete train, validation, and test sets, we used: [LUAD & LUSC datasets](https://drive.google.com/drive/folders/18skVh8Vk6zoxG3Se5Vlb7a3EKP2xHXXd)
3. To get original HPL tile embeddings, we used: [LUAD & LUSC tile vector representations](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view?usp=sharing)
4. To get the original HPL-HPC assignments, we used: [LUAD vs LUSC type classification and HPC assignments](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht)

## Modifications

### HPL-VICReg
*Point person: Jennifer Motter*

Original VICReg paper: https://arxiv.org/pdf/2105.04906

We changed the self-supervised learning (SSL) method of HPL from Barlow Twins to Variance-Invariance-Covariance Regularization (VICReg).

### HPL-ViT
*Point person: Alyssa Pradhan*

Original ViT paper: https://arxiv.org/pdf/2010.11929

Details: 

We replaced the convolutional neural network (CNN) backbone of HPL to a vision transformer (ViT).

### HPL-CLIP
*Point person: Yumi Briones*

Original CLIP paper: https://arxiv.org/pdf/2103.00020

Details: https://github.com/yumibriones/HPL-Modified/blob/main/CLIP/README.md

To enable multimodal learning, we integrated Contrastive Language-Image Pre-Training (CLIP) by OpenAI ([open_clip implementation](https://github.com/mlfoundations/open_clip)) into the HPL pipeline.

#### HPL-CONCH

As a bonus, we generated and clustered image embeddings from CONtrastive learning from Captions for Histopathology (CONCH) by the Mahmood Lab (https://github.com/mahmoodlab/CONCH). This is a CLIP-style model that has been trained on over a million histopathology image-caption pairs. A caveat is that pathological information is included in the captions, so clusters generated from this method will not be completely unsupervised.

## Results

We redid UMAP and Leiden clustering on the original HPL embeddings. We repeated this analysis for all modifications of HPL (i.e., HPL-CLIP, HPL-CONCH, HPL-VICReg, HPL-ViT). Results can be found here: [HPL-Modified Results](https://drive.google.com/drive/folders/11N90nfzHcVXhI4aQpWc3PjFSY3ryGdMr?usp=sharing).

Briefly, this is how results were generated:

1. Extract embeddings from the original HPL pipeline using [extract_embeddings_hpl.ipynb](https://github.com/yumibriones/HPL-Modified/blob/main/notebooks/extract_embeddings_hpl.ipynb).
2. Extract embeddings from HPL-CLIP using [extract_embeddings_clip.py](https://github.com/yumibriones/HPL-Modified/blob/main/scripts/py/extract_embeddings_clip.py).*
3. Extract embeddings from HPL-CONCH using [extract_embeddings_conch.py](https://github.com/yumibriones/HPL-Modified/blob/main/scripts/py/extract_embeddings_conch.py).*
4. Run UMAP/Leiden clustering on embeddings using [run_umap_leiden.py](https://github.com/yumibriones/HPL-Modified/blob/main/scripts/py/run_umap_leiden.py).*
5. Plot UMAP with clustering results/clinical features overlaid on top using [plot_umap.py](https://github.com/yumibriones/HPL-Modified/blob/main/scripts/py/plot_umap.py).*

*If submitting as a batch job on HPC, use corresponding scripts in each folder: VICReg, ViT, CLIP. Make sure to adjust filepaths accordingly.

## Evaluation

We evaluated our models in terms of (1) similarity of clusters to the original HPL pipeline, and (2) how well the clusters separate LUAD from LUSC. Evaluation is done here: [evaluation.ipynb](https://github.com/yumibriones/HPL-Modified/blob/main/notebooks/evaluation.ipynb).

