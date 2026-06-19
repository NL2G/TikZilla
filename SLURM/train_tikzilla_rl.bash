#!/bin/bash -l
#SBATCH --job-name=train_tikzilla_rl
#SBATCH --output=logs/train_tikzilla_rl_%A_%a.out
#SBATCH --error=logs/train_tikzilla_rl_%A_%a.err
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

conda activate /home/hpc/<USERNAME>/tikzilla_rl

export WORKDIR="$(ws_find tikzilla-training)"

accelerate launch --config_file Accelerate_Configs/deepspeed_rl.yaml Training/train_tikzilla_rl.py --work_dir "$WORKDIR" --tmp_dir "$TMPDIR"