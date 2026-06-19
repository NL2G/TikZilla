import os
import json
import random
import tarfile

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import concatenate_datasets
from datasets import Dataset as HFDataset


def parse_sources(source_filter):
    valid_sources = {"arxiv", "github", "tex", "synthetic", "curated"}
    requested = set(source_filter.split("_"))
    unknown = requested - valid_sources
    if unknown:
        raise ValueError(f"Unknown source(s): {unknown}")
    return requested


def construct_prompt(entry, input_variant, all_entries_old_caption):
    if input_variant == "caption":
        caption = entry.get("caption", "").strip()
        if not caption:
            return "", all_entries_old_caption
        all_entries_old_caption += 1
        return entry.get("caption", ""), all_entries_old_caption
    elif input_variant == "text_mentions":
        return " ".join(entry.get("text_mentions", []))
    elif input_variant == "caption_text_mentions":
        caption = entry.get("caption", "")
        mentions = " ".join(entry.get("text_mentions", []))
        return f"{caption} {mentions}".strip()
    elif input_variant == "new_caption":
        return entry.get("new_caption", "")
    elif input_variant == "caption_or_new_caption":
        caption = entry.get("caption", "")
        if caption.strip():
            return caption
        return entry.get("new_caption", "")
    elif input_variant == "caption_and_new_caption":
        return None
    elif input_variant == "new_caption_comparison":
        caption = entry.get("caption", "").strip()
        if not caption:
            return ""
        return entry.get("new_caption", "")
    return ""


def get_huggingface_dataset(json_dir, input_variant, code_length, source_filter, number_samples, data_percentage, relative, compiled, debugged, seed):
    json_paths = [
        os.path.join(json_dir, fname)
        for fname in os.listdir(json_dir)
        if fname.startswith("all_") and fname.endswith(".json")
    ]
    if debugged:
        new_json_dir = json_dir.replace("/all", "/all_new")
        debugged_json_paths = [
            os.path.join(new_json_dir, fname)
            for fname in os.listdir(new_json_dir)
            if fname.startswith("all_new_") and fname.endswith(".json")
        ]
    else:
        debugged_json_paths = []
    all_paths = json_paths + debugged_json_paths
    allowed_sources = parse_sources(source_filter)
    min_len, max_len = code_length
    data = []
    all_entries = 0
    all_entries_old_caption = 0
    for path in all_paths:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                all_entries += 1
                if compiled:
                    if entry.get("status") != "success":
                        continue
                if entry.get("source") not in allowed_sources:
                    continue
                cl = entry.get("code_length", 0)
                if not (min_len <= cl <= max_len):
                    continue

                if input_variant == "caption_and_new_caption":
                    caption = entry.get("caption", "").strip()
                    new_caption = entry.get("new_caption", "").strip()
                    if caption:
                        data.append({"text": caption, "label": entry["code"]})
                    if new_caption:
                        data.append({"text": new_caption, "label": entry["code"]})
                else:
                    if input_variant == "caption":
                        prompt, all_entries_old_caption = construct_prompt(entry, input_variant, all_entries_old_caption)
                    else:
                        prompt = construct_prompt(entry, input_variant, all_entries_old_caption)
                    if not prompt.strip():
                        continue
                    data.append({"text": prompt, "label": entry["code"]})
    total = len(data)
    if relative:
        target = int(total * data_percentage)
    else:
        target = number_samples
    if target < total:
        random.seed(seed)
        data = random.sample(data, target)
    return HFDataset.from_list(data)


def process_metadata_tar(json_path, tar_path, code_length):
    examples = []
    try:
        with open(json_path, "r") as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        return examples
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            images = {}
            for m in tar.getmembers():
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                file_id = os.path.basename(m.name).replace(".png", "")
                try:
                    img_bytes = tar.extractfile(m).read()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    images[file_id] = image
                except Exception as e:
                    continue
            for entry in entries:
                tikz_code = entry.get("code")
                file_id = entry.get("file_id")
                if (
                    tikz_code is None
                    or file_id is None
                    or not (code_length[0] <= len(tikz_code) <= code_length[1])
                    or file_id not in images
                ):
                    continue
                examples.append({
                    "text": tikz_code,
                    "image": images[file_id]
                })
    except Exception as e:
        pass
    return examples


def save_huggingface_dataset_chunks_streamed(json_dir, code_length, tmp_dir):
    output_dir = os.path.join(tmp_dir, "chunks")
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(tmp_dir, "unified_dataset", "images")
    json_files = sorted([
        fname for fname in os.listdir(json_dir)
        if fname.startswith("metadata_") and fname.endswith(".json")
    ])
    chunk_paths = []
    for idx, fname in tqdm(list(enumerate(json_files)), desc="Streaming tar+json to disk"):
        archive_id = fname[len("metadata_"):-len(".json")]
        json_path = os.path.join(json_dir, fname)
        tar_path = os.path.join(image_dir, f"images_{archive_id}.tar.gz")
        if not os.path.exists(tar_path):
            continue
        try:
            examples = process_metadata_tar(json_path, tar_path, code_length)
            if not examples:
                continue
            dataset = HFDataset.from_list(examples)
            chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}")
            dataset.save_to_disk(chunk_path)
            chunk_paths.append(chunk_path)
        except Exception as e:
            continue
    return chunk_paths


def load_saved_chunks(chunk_paths):
    datasets = [HFDataset.load_from_disk(path) for path in chunk_paths]
    return concatenate_datasets(datasets)