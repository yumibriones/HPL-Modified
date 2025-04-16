#!/bin/bash
#SBATCH --partition=fn_short                  # Node
#SBATCH --nodes=1                             # 1 node
#SBATCH --job-name=sample_py_job         # Job name - REPLACE WITH YOUR JOB NAME
#SBATCH --output=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/%x_%j.out  # Redirect stdout log to logs directory
#SBATCH --error=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs%x_%j.err   # Redirect stderr log to logs directory
#SBATCH --cpus-per-task=1                     # Run on a single CPU
#SBATCH --mem-per-cpu=50GB                     # Memory limit
#SBATCH --time=12:00:00                       # Max time for short

# MODIFY ABOVE CONFIGURATIONS AS DESIRED

# activate conda venv
source /gpfs/data/hammelllab/yumi/lib/yumi_miniconda/etc/profile.d/conda.sh  # REPLACE WITH PATH TO YOUR CONDA ENV
conda activate dl4med_25  # REPLACE WITH NAME OF YOUR CONDA ENV
echo "conda activated"

echo "Using Python at: $(which python)"

# Navigate to the directory where your py script is located
cd /gpfs/home/yb2612/dl4med_25/dl_project/bin/py  # THIS SHOULD WORK

# run py script with live updates (-u)
python -u [SCRIPT].py  # REPLACE WITH NAME OF YOUR SCRIPT