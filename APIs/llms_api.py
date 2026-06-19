import os
import json
import torch
import openai

from contextlib import contextmanager
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


@contextmanager
def temporary_change_attributes(obj, **temporary_values):
    original_values = {k: getattr(obj, k) for k in temporary_values}
    for k, v in temporary_values.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in original_values.items():
            setattr(obj, k, v)


class Gpt4Api:
    def __init__(self, model_id, temperature=1.0, top_p=0.9, max_new_tokens=2048):
        with open("<YOUR_API_KEY>.json", "r") as file:
            api_key = json.load(file)
        self.client = openai.OpenAI(api_key=api_key["api_key"])
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def request(self, user_prompt):
        completion = self.client.chat.completions.create(
        model=self.model_id,
        temperature=self.temperature,
        top_p=self.top_p,
        max_tokens=self.max_new_tokens,
        messages=[
            {
                "role": "user", "content": 
                "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                + user_prompt +
                "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                "Only output valid LaTeX code with no extra text."
            }
        ]
        )
        return completion.choices[0].message.content


class Gpt5Api:
    def __init__(self, model_id, temperature=1.0, top_p=0.9, max_new_tokens=2048):
        with open("<YOUR_API_KEY>.json", "r") as file:
            api_key = json.load(file)
        self.client = openai.OpenAI(api_key=api_key["api_key"])
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def request(self, user_prompt):
        completion = self.client.chat.completions.create(
        model=self.model_id,
        messages=[
            {
                "role": "user", "content": 
                "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                + user_prompt +
                "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                "Only output valid LaTeX code with no extra text."
            }
        ]
        )
        return completion.choices[0].message.content


class Llama3_1Api:
    def __init__(self, work_dir, model_id, temperature=1.0, top_p=0.9, max_new_tokens=2048):
        base_model_path = f"{work_dir}/models/{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        vocab = self.tokenizer.get_vocab()
        self.eot_id = (self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in vocab else self.tokenizer.eos_token_id)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def request(self, user_prompt):
        messages = [
            {"role": "user", "content": 
            "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
            + user_prompt +
            "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
            "Only output valid LaTeX code with no extra text."
            }
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([rendered], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature is not None and self.temperature > 0.0),
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.eot_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text


class Qwen3CoderApi:
    def __init__(self, work_dir, model_id, temperature=1.0, top_p=0.9, max_new_tokens=2048):
        base_model_path = f"{work_dir}/models/{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def request(self, user_prompt):
        messages = [
            {
                "role": "user",
                "content": (
                    "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                    + user_prompt +
                    "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                    "Only output valid LaTeX code with no extra text."
                )
            }
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0, prompt_len:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


class QwenApi:
    def __init__(self, work_dir, model_id, adapter_path, finetuned, temperature=1.0, top_p=0.9, merge_lora=True, max_new_tokens=2048, device_map="auto"):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if model_id.endswith("-finetuned-lora"):
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Missing adapter_config.json at {adapter_path}")
            with open(config_path, "r") as f:
                lora_config_dict = json.load(f)

            base_model_path = lora_config_dict.get("base_model_name_or_path")
            if base_model_path is None:
                raise ValueError("adapter_config.json must include 'base_model_name_or_path'.")

        elif model_id.endswith("-finetuned-full"):
            base_model_path = adapter_path

        else:
            base_model_path = f"{work_dir}/models/{model_id}"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id

        if finetuned:
            self.gen_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
        else:
            self.gen_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=self.max_new_tokens
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )

        if finetuned:
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token

            if model_id.endswith("-finetuned-lora"):
                lora_config = LoraConfig(**lora_config_dict)

                if merge_lora:
                    with temporary_change_attributes(torch.cuda, is_available=lambda: False):
                        self.model = PeftModel.from_pretrained(
                            base_model,
                            adapter_path,
                            config=lora_config,
                            torch_dtype=torch.bfloat16
                        ).merge_and_unload()
                else:
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        adapter_path,
                        config=lora_config,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map
                    )
            else:
                self.model = base_model
        else:
            self.model = base_model

        self.model.eval()

    def request(self, user_prompt):
        messages = [
            {"role": "user", "content": (
                "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                + user_prompt +
                "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                "Only output valid LaTeX code with no extra text."
            )}
        ]
        if self.model_id == "Qwen3-32B":
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = self.tokenizer([full_text], return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(
            **inputs,
            generation_config=self.gen_config
        )[0][len(inputs["input_ids"][0]):].tolist()
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)