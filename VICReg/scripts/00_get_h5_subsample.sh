#!/bin/bash
#SBATCH --partition=gpu8_long,gpu4_long
#SBATCH --job-name=00_split_training_250k
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Jennifer.Motter@nyulangone.org
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=00_split_training_250k_%A_%a.out
#SBATCH --error=00_split_training_250k_%A_%a.err
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1

#Variables
num_samples=250000

# Define paths
path_to_code="/gpfs/data/pmedlab/Users/mottej02/dl_project/pipeline/Histomorphological-Phenotype-Learning/utilities/h5_handling/subsample_h5.py"     
h5_file="/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train-002.h5"  
output_dir="/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/"
output_file="/gpfs/scratch/yb2612/dl4med_25/dl_project/scratch_data/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_train.h5"

#mkdir $output_dir
              
# Print information 
echo "Starting H5 file subsampling"
echo "Source file: $h5_file"
echo "Number of samples: $num_samples"
echo "Using script: $path_to_code"
echo "-----------------------"

# Set environment variable to avoid HDF5 file locking issues
export HDF5_USE_FILE_LOCKING=FALSE

module load condaenvs/gpu/pathgan_SSL

# Run the Python script
python $path_to_code \
    --h5_file $h5_file \
    --num_samples $num_samples 

generated_output="hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_train.h5"


# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Subsampling completed successfully!"
else
    echo "Error: Subsampling failed!"
fi


# https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/README_HPL.md#Workspace-setup
