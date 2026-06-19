#!/bin/bash -l
#SBATCH --job-name=debug_tikz
#SBATCH --output=logs/debug_tikz_%A_%a.out
#SBATCH --error=logs/debug_tikz_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python/3.12-conda

conda activate /home/hpc/<USERNAME>/tikzilla_sft

export WORKDIR="$(ws_find tikzilla-training)"

MODEL_ID="Qwen3-32B"
MODEL_PATH="$WORKDIR/models/$MODEL_ID"
DATASET_PATH="$WORKDIR/debug_prompts"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --max-num-seqs 128 \
    --host 0.0.0.0 \
    --port 8000 > /dev/null 2>&1 &

while ! curl -s http://localhost:8000/v1/models >/dev/null; do
    sleep 2
done

python LLM_Debug/debug_tikz.py \
    --model_path "$MODEL_PATH" \
    --batch_size 128 \
    --dataset_path "$DATASET_PATH" \
    --max_retries 3