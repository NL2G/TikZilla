import os
import re
import json
import asyncio
import aiohttp
import argparse

from pathlib import Path
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio


API_KEY = "no-key-needed"
API_BASE = "http://localhost:8000/v1"
CLIENT = OpenAI(base_url=API_BASE, api_key=API_KEY)


STATIC_PROMPT = (
    "You are a TikZ expert. I will provide TikZ code and the corresponding LaTeX error log. "
    "Fix the TikZ code so it compiles without errors. Only output the corrected TikZ code."
)


def arg_parser():
    parser = argparse.ArgumentParser(description="Debug Tikz figures using LLMs.")
    parser.add_argument('--model_path', type=str, required=True, help="The model path to use.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for processing prompts.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--max_retries', type=int, required=True, help="Maximum number of retries for failed requests.")
    return parser.parse_args()


def extract_tikz_from_response(response):
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    first = response.find('\\')
    last = response.rfind('\\')
    end = response.find('\n', last)
    return response[first:end].strip() if first != -1 and end != -1 else response.strip()


def read_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def call_model(session, model_path, prompt, file_id, max_retries):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. /nothink"},
        {"role": "user", "content": STATIC_PROMPT + "\n\n" + prompt}
    ]
    payload = {
        "model": model_path,
        "messages": messages
    }
    url = f"{API_BASE}/chat/completions"
    for _ in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return {
                        "file_id": file_id,
                        "response": extract_tikz_from_response(content)
                    }
                else:
                    continue
        except Exception:
            continue
        await asyncio.sleep(2)
    return None


async def process_all(args, prompts, idx):
    chunk_outputs = []
    connector = aiohttp.TCPConnector(limit=args.batch_size)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            call_model(session, args.model_path, item["prompt"], item["file_id"], args.max_retries)
            for item in prompts
        ]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
            output = await coro
            if output is None:
                continue
            chunk_outputs.append(output)
    chunk_file = Path("outputs") / f"debug_{idx}.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for item in chunk_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = arg_parser()
    filenames = sorted([f for f in os.listdir(args.dataset_path) if f.endswith(".jsonl")])
    for idx, filename in enumerate(filenames):
        json_path = os.path.join(args.dataset_path, filename)
        prompts = read_prompts(json_path)
        asyncio.run(process_all(args, prompts, idx))