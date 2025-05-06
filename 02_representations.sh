#!/bin/bash
#SBATCH --partition=gpu8_long,gpu4_long
#SBATCH --job-name=02_run_representationspathology_projection_Barlow3_27
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Jennifer.Motter@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=02_run_representationspathology_projection_Barlow3_27_%A_%a.out
#SBATCH --error=02_run_representationspathology_projection_Barlow3_27_%A_%a.err
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2

#vars
px_size=224
mag=5
z_dim=128
epoch=27
model="BarlowTwins_3"

#paths
path_to_code="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/run_representationspathology_projection.py"
path_to_checkpoint="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/data_model_output/${model}/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/results/epoch_${epoch}/checkpoints/BarlowTwins_3.ckt"
output_hdf5_dir="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/datasets/TCGAFFPE_LUADLUSC_5x_60pc_250K/he/patches_h${px_size}_w${px_size}"
output_hdf5_train="${output_hdf5_dir}/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_train.h5"
output_hdf5_valid="${output_hdf5_dir}/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_validation.h5"
output_hdf5_test="${output_hdf5_dir}/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_test.h5"

module load pathganplus/3.6
python3 $path_to_code \
	--checkpoint $path_to_checkpoint \
	--real_hdf5 $output_hdf5_train \
	--model BarlowTwins_3 \
	--dataset TCGAFFPE_LUADLUSC_5x_60pc_250K \
	--img_size $px_size

python3 $path_to_code \
        --checkpoint $path_to_checkpoint \
        --real_hdf5 $output_hdf5_valid \
        --model BarlowTwins_3 \
        --dataset TCGAFFPE_LUADLUSC_5x_60pc_250K \
        --img_size $px_size

python3 $path_to_code \
        --checkpoint $path_to_checkpoint \
        --real_hdf5 $output_hdf5_test \
        --model BarlowTwins_3 \
        --dataset TCGAFFPE_LUADLUSC_5x_60pc_250K \
        --img_size $px_size

#GitHub: https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning


