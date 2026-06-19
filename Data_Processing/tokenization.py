import copy
from transformers import AutoTokenizer


def make_qwen_base_preprocessor(model_path, max_seq_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    def preprocess_batch(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for desc, code in zip(batch["text"], batch["label"]):
            desc = desc.strip()
            code = code.strip()

            if not code.endswith(r"\end{document}"):
                continue

            desc_token_count = len(tokenizer(desc, add_special_tokens=False)["input_ids"])
            if desc_token_count > int(max_seq_len / 4):
                continue

            messages = [
                {"role": "user", "content": (
                    "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                    + desc +
                    "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                    "Only output valid LaTeX code with no extra text."
                )},
                {"role": "assistant", "content": code}
            ]

            try:
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                continue

            tokenized = tokenizer(
                full_text, 
                add_special_tokens=False
            )
            input_ids = tokenized["input_ids"]

            if len(input_ids) >= max_seq_len:
                continue
            
            attention_mask = [1] * len(input_ids)

            prompt_messages = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt_token_count = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            labels = [-100] * prompt_token_count + input_ids[prompt_token_count:]
            assert len(labels) == len(input_ids)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    return preprocess_batch


def make_qwen_base_preprocessor_grpo(model_path, max_seq_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    def preprocess_batch(batch):
        prompts = []
        ground_truths = []
        reward_signals = []

        for desc, code, image in zip(batch["text"], batch["code"], batch["image"]):
            desc = desc.strip()
            code = code.strip()

            if not code.endswith(r"\end{document}"):
                continue

            messages = [
                {"role": "user", "content": (
                    "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                    + desc +
                    "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                    "Only output valid LaTeX code with no extra text."
                )}
            ]

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                continue

            text_tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].shape[-1]
            completion_tokens = tokenizer(code, truncation=False, return_tensors="pt")["input_ids"].shape[-1]
            if text_tokens > int(max_seq_len / 4) or completion_tokens > max_seq_len:
                continue

            prompts.append(text)
            ground_truths.append(code)
            reward_signals.append(image)

        return {
            "prompt": prompts,
            "ground_truth": ground_truths,
            "reward_signal": reward_signals,
        }

    return preprocess_batch


def tokenize_detikzify(batch, processor, **kwargs):
    image_token = processor.image_token
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    input_ids = processor(
        text=batch['text'],
        images=batch['image'],
        max_length=processor.tokenizer.model_max_length,
        pad_to_multiple_of=8,
        add_eos_token=True,
        **kwargs
    )
    input_ids['labels'] = copy.deepcopy(input_ids['input_ids'])
    for label_ids in input_ids['labels']:
        for idx, label_id in enumerate(label_ids):
            if label_id in {image_token_id, processor.tokenizer.pad_token_id}:
                label_ids[idx] = -100
    return input_ids