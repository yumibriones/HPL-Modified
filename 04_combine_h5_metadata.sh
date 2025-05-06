#!/bin/bash
#SBATCH --partition=gpu4_short
#SBATCH --job-name=04_combining_h5_metadata_BarlowTwins3_27
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Jennifer.Motter@nyulangone.org
#SBATCH --cpus-per-task=20
#SBATCH --output=04_combining_h5_metadata_BarlowTwins3_27_%A_%a.out
#SBATCH --error=04_combining_h5_metadata_BarlowTwins3_27_%A_%a.err
#SBATCH --mem=5GB
#SBATCH --time=05:00:00

## TRAIN, VALIDATION, AND TEST SETS
## Number of folds will determine Leiden clustering later

#vars
epoch=27
model="BarlowTwins_3"

#paths
path_to_code="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/utilities/h5_handling/create_metadata_h5.py"
path_to_metadata="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/metadata/LUADLUSC_lungsubtype_overall_survival.csv"
path_to_h5="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/results/${model}/epoch_${epoch}/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_complete.h5"

module load pathganplus/3.6
python3 $path_to_code \
    --meta_file $path_to_metadata \
    --meta_name TCGAFFPE_LUADLUSC_5x_60pc_250K_metadata \
    --matching_field slides \
    --list_meta_field luad \
    --h5_file $path_to_h5	
    
#GitHub: https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning
