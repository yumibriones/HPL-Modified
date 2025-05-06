#!/bin/bash
#SBATCH --partition=gpu4_long,gpu8_long
#SBATCH --job-name=03_combine_projections_BarlowTwins_3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Jennifer.Motter@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=03_combine_projections_BarlowTwins_3_%A_%a.out
#SBATCH --error=03_combine_projections_BarlowTwins_3_%A_%a.err
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2

#vars
px_size=224
z_dim=128

#paths
path_to_code="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/utilities/h5_handling/combine_complete_h5.py"
main_path="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning" 

module load pathganplus/3.6
python3 $path_to_code \
	    --img_size $px_size \
	    --z_dim $z_dim \
	    --dataset TCGAFFPE_LUADLUSC_5x_60pc_250K \
	    --model BarlowTwins_3 \
        --main_path $main_path

