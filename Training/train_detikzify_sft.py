import os
import torch
import argparse

from functools import partial
from Data_Processing.tokenization import tokenize_detikzify
from detikzify.model import load as load_model
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets, load_from_disk
from transformers.trainer_utils import get_last_checkpoint

def arg_parser():
    parser = argparse.ArgumentParser(description="Finetune LLMs on TikZ code (inverse graphics).")
    parser.add_argument('--model_id', type=str, default="detikzify-v2-8b", help="ID of the LLM to be finetuned.")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length of input + output.")
    parser.add_argument('--code_length', type=tuple, default=(100, 4000), help="Min and max TikZ code lengths.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for finetuning.")
    parser.add_argument('--device_batch_size', type=int, default=16, help="Batch size per device.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number steps for gradient accumulation.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for finetuning.")
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Value for gradient clipping.")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup percentage of sheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate sheduler type.")
    parser.add_argument('--work_dir', type=str, required=True, help="Path to work dir.")
    parser.add_argument('--tmp_dir', type=str, required=True, help="Path to tmp dir.")
    return parser.parse_args()

def main():
    args = arg_parser()

    detikzify_model_id = f"{args.model_id}_{args.max_seq_length}_{args.code_length[0]}_{args.code_length[1]}"

    model_path = os.path.join(args.work_dir, "models", args.model_id)
    model, processor = load_model(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    save_dir = os.path.join(args.work_dir, "processed_dataset_detikzify")
    shard_paths = sorted(
        [os.path.join(save_dir, d) for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))],
        key=lambda x: int(x.split("_")[-1])
    )
    if not shard_paths:
        raise FileNotFoundError(f"No .arrow files found in {save_dir}")
    datasets_list = [load_from_disk(p) for p in shard_paths]
    full_dataset = concatenate_datasets(datasets_list)
    tokenizer_path = os.path.join(args.work_dir, "tokenizer_detikzify", f"{detikzify_model_id}.arrow")

    subset = full_dataset.shuffle(seed=args.seed)
    processed_dataset = subset.map(
        partial(tokenize_detikzify, processor=processor, truncation=True, padding="max_length"),
        batched=True,
        batch_size=256,
        num_proc=32,
        remove_columns=["image", "text"],
        load_from_cache_file=True,
        cache_file_name=tokenizer_path,
        desc="Tokenizing",
    )

    trained_model_path = os.path.join(args.work_dir, "trained_models_detikzify", detikzify_model_id)
    sft_config = TrainingArguments(
        output_dir=trained_model_path,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.device_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
		logging_dir=f"tf_logs_detikzify/{detikzify_model_id}",
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=sft_config,
        train_dataset=processed_dataset,
    )

    last_checkpoint = None
    if os.path.isdir(trained_model_path):
        last_checkpoint = get_last_checkpoint(trained_model_path)
    trainer.train(resume_from_checkpoint=last_checkpoint)

if __name__=="__main__":
    main()