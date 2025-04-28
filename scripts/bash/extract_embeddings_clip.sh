#!/bin/bash
#SBATCH --partition=gpu8_short                  # Node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB                     # Memory limit
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=extract_embeddings_clip         # Job name
#SBATCH --output=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/%x_%j.out  # Redirect stdout log to logs directory
#SBATCH --error=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/%x_%j.err   # Redirect stderr log to logs directory
#SBATCH --time=12:00:00                       # Max time for short

# activate conda venv
source /gpfs/data/hammelllab/yumi/lib/yumi_miniconda/etc/profile.d/conda.sh
conda activate dl4med_25
echo "conda activated"

echo "Using Python at: $(which python)"

# navigate to cloned open_clip repo
cd /gpfs/home/yb2612/dl4med_25/dl_project/scripts/py

# run py script with live updates (-u)
python -u extract_embeddings_clip.py