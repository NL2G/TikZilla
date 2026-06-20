import os
import json
import tarfile
import argparse

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    return parser.parse_args()


def process_metadata_tar(json_path, tar_path, code_length):
    examples = []
    try:
        with open(json_path, "r") as f:
            entries = json.load(f)
    except json.JSONDecodeError:
        return examples
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            images = {}
            for m in tar.getmembers():
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                file_id = os.path.basename(m.name).replace(".png", "")
                try:
                    with tar.extractfile(m) as f:
                        img_bytes = f.read()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    images[file_id] = image
                except Exception:
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
    except Exception:
        pass
    return examples


def save_huggingface_dataset_chunks_streamed(json_dir, image_dir, code_length, output_dir, batch_size=512):
    os.makedirs(output_dir, exist_ok=True)
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
            with open(json_path, "r") as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            continue
        meta_dict = {
            entry["file_id"]: entry["code"]
            for entry in entries
            if "file_id" in entry and "code" in entry
               and code_length[0] <= len(entry["code"]) <= code_length[1]
        }
        examples = []
        try:
            dataset = None
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not member.name.endswith(".png"):
                        continue
                    file_id = os.path.basename(member.name).replace(".png", "")
                    if file_id not in meta_dict:
                        continue
                    try:
                        img_bytes = tar.extractfile(member).read()
                        image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        examples.append({"text": meta_dict[file_id], "image": image})
                    except Exception:
                        continue
                    if len(examples) >= batch_size:
                        batch_dataset = Dataset.from_list(examples)
                        if dataset is None:
                            dataset = batch_dataset
                        else:
                            dataset = concatenate_datasets([dataset, batch_dataset])
                        examples.clear()
            if examples:
                batch_dataset = Dataset.from_list(examples)
                if dataset is None:
                    dataset = batch_dataset
                else:
                    dataset = concatenate_datasets([dataset, batch_dataset])
                examples.clear()
            if dataset is not None:
                chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}")
                dataset.save_to_disk(chunk_path)
                chunk_paths.append(chunk_path)
                del dataset
        except Exception:
            continue
    return chunk_paths


def load_saved_chunks(chunk_paths):
    datasets_list = [load_from_disk(path) for path in chunk_paths]
    return concatenate_datasets(datasets_list)


if __name__ == "__main__":
    code_length = (100, 4000)
    save_dir = "datikz/processed_dataset_detikzify"
    os.makedirs(save_dir, exist_ok=True)
    chunk_paths = sorted(
        [os.path.join(save_dir, f) for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
    )
    if chunk_paths:
        full_dataset = load_saved_chunks(chunk_paths)
    else:
        json_dir = "datikz/unified_dataset/metadata"
        image_dir = "datikz/unified_dataset/images"
        chunk_paths = save_huggingface_dataset_chunks_streamed(
            json_dir=json_dir,
            image_dir=image_dir,
            code_length=code_length,
            output_dir=save_dir
        )
        full_dataset = load_saved_chunks(chunk_paths)
    num_shards = 10
    for idx in range(num_shards):
        shard = full_dataset.shard(num_shards=num_shards, index=idx)
        shard_path = os.path.join(save_dir, f"shard_{idx:02d}")
        shard.save_to_disk(shard_path)