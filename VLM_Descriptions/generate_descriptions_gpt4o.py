import os
import json
import base64

from tqdm import tqdm
from pathlib import Path
from APIs.llms_api import GptApi


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


def read_image_paths(dataset_path):
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(dataset_path)) if f.lower().endswith('.png')]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_images(api, base64_encoded_images, image_basenames, idx):
    chunk_outputs = []
    for image_b64, file_id in tqdm(zip(base64_encoded_images, image_basenames), total=len(image_basenames), desc="Processing images"):
        try:
            result = api.request_with_images(STATIC_PROMPT, image_b64)
            chunk_outputs.append({
                "file_id": file_id,
                "new_caption": result.strip()
            })
        except Exception:
            continue
    chunk_file = Path("outputs") / f"spiqa_descriptions_{idx}.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for item in chunk_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    api = GptApi(system_prompt=STATIC_PROMPT, model_id="gpt-4o")
    for idx in range(1):
        dataset_path = "spiqa/images"
        image_basenames = read_image_paths(dataset_path)
        base64_encoded_images = [encode_image(os.path.join(dataset_path, f"{name}.png")) for name in image_basenames]
        process_images(api, base64_encoded_images, image_basenames, idx)