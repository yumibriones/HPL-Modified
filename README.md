# HPL-Modified

Modifications to the [Histomorphological Phenotype Learning pipeline (Quiros et al.)](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning). For our Spring 2025 Deep Learning for Medicine class.

## Contact information
Yumi Briones - yb2612@nyu.edu, Yumi.Briones@nyulangone.org  
Jennifer Motter - mottej02@nyu.edu, Jennifer.Motter@nyulangone.org  
Alyssa Pradhan - amp10295@nyu.edu, Alyssa.Pradhan@nyulangone.org  

## Repo structure
* `docs` - documentation
* `scripts` - scripts for automation (bash scripts go here)
* `src` - source code
* `tests` - test runs, Jupyter notebooks, etc.

## About the data
To follow.

## HPL-CLIP
*Point person: Yumi Briones*

This integrates Contrastive Language-Image Pre-Training (CLIP) by OpenAI (specifically the [open_clip](https://github.com/mlfoundations/open_clip) implementation) into HPL. Specifically, we replace the CNN backbone of HPL with a ViT-B-32 architecture from open_clip.

![image](HPL-CLIP_diagram.png)

Tutorial for running HPL-CLIP: https://github.com/yumibriones/HPL-Modified/blob/main/docs/HPL-CLIP_tutorial.md

## HPL-VICReg
*Point person: Jennifer Motter*

## HPL-ViT
*Point person: Alyssa Pradhan*
