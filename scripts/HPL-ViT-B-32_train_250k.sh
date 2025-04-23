#!/bin/bash
#SBATCH --partition=gpu8_medium
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20GB
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=HPL-ViT-B-32_train_250k
#SBATCH --output=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/%x_%j.out  # Redirect stdout log to logs directory
#SBATCH --error=/gpfs/home/yb2612/dl4med_25/dl_project/results/logs/%x_%j.err   # Redirect stderr log to logs directory
#SBATCH --time=48:00:00

# activate conda venv
source /gpfs/data/hammelllab/yumi/lib/yumi_miniconda/etc/profile.d/conda.sh
conda activate dl4med_25
echo "conda activated"

echo "Using Python at: $(which python)"

# navigate to cloned open_clip repo
cd /gpfs/home/yb2612/dl4med_25/dl_project/open_clip

export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="/gpfs/home/yb2612/dl4med_25/dl_project/data/lung_250k_filepath_caption.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --logs /gpfs/home/yb2612/dl4med_25/dl_project/results/logs \
    --warmup 2000 \
    --batch-size=256 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=4 \
    --model ViT-B-32 \
    --name "HPL-ViT-B-32_train_250k" \
    --seed 9 \
    --local-loss \
    --gather-with-grad