#!/bin/bash
#SBATCH --partition=gpu8_long,gpu4_long
#SBATCH --job-name=01_run_representationspathology_VICReg_0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Jennifer.Motter@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=01_run_representationspathology_VICReg_0_%A_%a.out
#SBATCH --error=01_run_representationspathology_VICReg_0_%A_%a.err
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2

##TEST, TRAIN AND VALIDATION SETS


#paths
path_to_code="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/run_representationspathology.py"
path_to_dataset="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/datasets/TCGAFFPE_LUADLUSC_5x_60pc_250K/he/patches_h224_w224"


#setting up folder structure and creating symbolic links
#mkdir $path_to_dataset
#mkdir $path_to_dataset/he
#mkdir $path_to_dataset/he/patches_h${px_size}_w${px_size}

cd $path_to_dataset

module purge
module load cuda/10.0
module load pathganplus/3.6

python3 $path_to_code \
	--img_size 224 \
	--batch_size 64 `#default size is 64` \
	--epochs 30 `#default is 40 epochs` \
	--z_dim 128 `#latent space size for constrastive loss` \
	--model VICReg_0 `#type of model (VICReg, BarlowTwins)` \
	--dataset TCGAFFPE_LUADLUSC_5x_60pc_250K \
	--check_every 1 `#save checkpoint and project samples every n epcohs` \
	--report `#report latent Space progress`

### NO PASS TWO FOR THIS STEP

#GitHub: https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning
