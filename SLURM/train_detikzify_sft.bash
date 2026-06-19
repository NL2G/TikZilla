#!/bin/bash -l
#SBATCH --job-name=train_detikzify_sft
#SBATCH --output=logs/train_detikzify_sft_%A_%a.out
#SBATCH --error=logs/train_detikzify_sft_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module purge
module load gcc/14.2.0
module load cuda/12.6.2
module load python/3.12-conda

conda activate /home/hpc/<USERNAME>/tikzilla_sft

export WORKDIR="$(ws_find tikzilla-training)"

accelerate launch --config_file Accelerate_Configs/deepspeed_sft.yaml Training/train_detikzify_sft.py --work_dir "$WORKDIR" --tmp_dir "$TMPDIR"