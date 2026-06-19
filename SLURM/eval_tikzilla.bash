#!/bin/bash -l
#SBATCH --job-name=eval_tikzilla
#SBATCH --output=logs/eval_tikzilla_%A_%a.out
#SBATCH --error=logs/eval_tikzilla_%A_%a.err
#SBATCH --time=01:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-5
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module purge
module load gcc/14.2.0
module load cuda/12.6.2
module load python/3.12-conda

conda activate /home/hpc/<USERNAME>/tikzilla_sft

export WORKDIR="$(ws_find tikzilla-training)"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Eval_Configs/eval_script_configs.txt)
IFS=',' read INPUT_FILE MODEL_ID <<< "$CONFIG"

python Evaluation/eval_tikzilla.py --input_file "$INPUT_FILE" --model_id "$MODEL_ID" --work_dir "$WORKDIR"