import os
import json
import tarfile
import base64
import asyncio
import aiohttp
import argparse

from pathlib import Path
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio


API_KEY = "no-key-needed"
API_BASE = "http://localhost:8000/v1"
CLIENT = OpenAI(base_url=API_BASE, api_key=API_KEY)
MAX_TOKENS = 1024


STATIC_PROMPT = (
    "You are a scientific illustrator describing images for precise redrawing in TikZ.\n"
    "Your task is to describe the image in precise, continuous prose without bullet points, lists, or line breaks.\n"
    "Start directly with the main object or scene. Avoid introductory phrases like 'Certainly!', 'The image depicts...', 'Here is a precise description.', ...\n"
    "Use clear, active language focused on geometry, labels, colors, spatial relationships, coordinates, and other visible properties.\n"
    "Describe all visible elements such as shapes, lines, arrows, and labels, including their relative or absolute positions, dimensions, and orientation.\n"
    "Use consistent, minimal naming for objects (e.g., 'circle A', 'line L1') and specify label positions relative to shapes precisely.\n"
    "Only describe exact, concrete visual elements that enable precise image reconstruction in TikZ.\n"
    "Avoid vague, interpretive, or inferential language, and exclude summaries, conclusions, or commentary about the image's meaning, function, or aesthetics.\n"
    "Here are a few examples:\n"
    "A thin black horizontal line centered in the middle, containing nine evenly spaced black dots, and labeled $x_2$ at the left. Each dot is connected by a thin black line in an alternating pattern to either $x_0$ (placed at the top middle) or $x_1$ (placed at the bottom middle).\n"
    "A line chart has different instruction scales of 1/10, 1/4, 1/2, and 1 on the x-axis. On the y-axis it shows BLEU scores between 20 and 50, with steps of 5. The chart contains three lines with Zh-En in blue, De-En in red, and Fr-En in brown. All BLEU scores are initially 20 at the lowest instruction scale. As the instruction scale increases, BLEU scores improve for all pairs. De-En is the highest, closely followed by Fr-En and then Zh-En far below. The increase is largest from 1/10 to 1/4 and only marginally above an instruction scale of 1/4. The legend is placed inside the chart at the top left.\n"
    "Write a description in this exact style for the given image."
)


def arg_parser():
    parser = argparse.ArgumentParser(description="Generate image descriptions using VLMs.")
    parser.add_argument('--model_path', type=str, required=True, help="The model path to use.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for processing prompts.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--max_retries', type=int, required=True, help="Maximum number of retries for failed requests.")
    return parser.parse_args()


def extract_tar_gz(tar_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def get_all_tarballs(dataset_path):
    return sorted([
        f for f in os.listdir(dataset_path)
        if f.endswith(".tar.gz")
    ])


def read_image_paths(dataset_path):
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(dataset_path)) if f.lower().endswith('.png')]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def call_model(session, model_path, base64_image, file_id, max_retries):
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": STATIC_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]
    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": MAX_TOKENS
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
                        "new_caption": content.strip()
                    }
                else:
                    continue
        except Exception:
            continue
        await asyncio.sleep(2)
    return None


async def process_all(args, base64_encoded_images, image_basenames, idx):
    chunk_outputs = []
    connector = aiohttp.TCPConnector(limit=args.batch_size)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            call_model(session, args.model_path, base64_image, basename, args.max_retries)
            for base64_image, basename in zip(base64_encoded_images, image_basenames)
        ]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
            output = await coro
            if output is None:
                continue
            chunk_outputs.append(output)
    chunk_file = Path("outputs") / f"descriptions_{idx}.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for item in chunk_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = arg_parser()
    tarballs = get_all_tarballs(args.dataset_path)
    for idx, tarball in enumerate(tarballs):
        tarball_path = os.path.join(args.dataset_path, tarball)
        tmp_folder_name = tarball.replace(".tar.gz", "") + "_tmp"
        extract_path = os.path.join(args.dataset_path, tmp_folder_name)
        extract_tar_gz(tarball_path, extract_path)
        debug_folder = tarball.replace(".tar.gz", "")
        extract_path_new = os.path.join(extract_path, debug_folder)
        images_dir = extract_path_new if os.path.isdir(extract_path_new) else extract_path
        image_paths = sorted([str(p) for p in Path(images_dir).rglob("*.png")])
        image_basenames = [Path(p).stem for p in image_paths]
        base64_encoded_images = [encode_image(p) for p in image_paths]
        asyncio.run(process_all(args, base64_encoded_images, image_basenames, idx))