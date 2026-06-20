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
    meta = {}
    for entry in entries:
        file_id = entry.get("file_id")
        code = entry.get("code")
        new_caption = entry.get("new_caption")
        if (
            file_id
            and isinstance(code, str)
            and isinstance(new_caption, str)
            and code_length[0] <= len(code) <= code_length[1]
            and new_caption.strip()
        ):
            meta[file_id] = {"code": code, "new_caption": new_caption}
    if not meta:
        return examples
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar.getmembers():
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                file_id = os.path.basename(m.name).replace(".png", "")
                if file_id not in meta:
                    continue
                try:
                    with tar.extractfile(m) as f:
                        img_bytes = f.read()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    continue
                examples.append({
                    "text": meta[file_id]["new_caption"],
                    "code": meta[file_id]["code"],
                    "image": image
                })
    except Exception:
        pass
    return examples


def save_huggingface_dataset_chunks_streamed(json_dir, image_dir, code_length, output_dir, start_archive, end_archive, batch_size=512):
    os.makedirs(output_dir, exist_ok=True)
    json_files = sorted([
        fname for fname in os.listdir(json_dir)
        if fname.startswith("metadata_") and fname.endswith(".json")
    ])
    if not json_files:
        return []
    chunk_paths = []
    for idx, fname in tqdm(list(enumerate(json_files)), desc="Streaming RL-quality tar+json to disk"):
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
            entry["file_id"]: (entry["code"], entry.get("new_caption", ""))
            for entry in entries
            if "file_id" in entry and "code" in entry
               and isinstance(entry["code"], str)
               and code_length[0] <= len(entry["code"]) <= code_length[1]
               and isinstance(entry.get("new_caption"), str)
               and entry.get("new_caption", "").strip()
        }
        if not meta_dict:
            continue
        examples = []
        dataset = None
        try:
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
                        code, new_caption = meta_dict[file_id]
                        examples.append({
                            "text": new_caption,
                            "code": code,
                            "image": image
                        })
                    except Exception:
                        continue
                    if len(examples) >= batch_size:
                        batch_dataset = Dataset.from_list(examples)
                        dataset = batch_dataset if dataset is None else concatenate_datasets([dataset, batch_dataset])
                        examples.clear()
            if examples:
                batch_dataset = Dataset.from_list(examples)
                dataset = batch_dataset if dataset is None else concatenate_datasets([dataset, batch_dataset])
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
    args = arg_parser()
    code_length = (100, 4000)
    save_dir = f"{args.work_dir}/datikz/processed_dataset_tikzilla_rl"
    os.makedirs(save_dir, exist_ok=True)
    force_rebuild = True
    if not force_rebuild:
        chunk_paths = sorted(
            [os.path.join(save_dir, f) for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f)) and f.startswith("chunk_")]
        )
    else:
        chunk_paths = []
    if chunk_paths:
        full_dataset = load_saved_chunks(chunk_paths)
    else:
        json_dir = f"{args.work_dir}/datikz/unified_dataset/metadata"
        image_dir = f"{args.work_dir}/datikz/unified_dataset/images"
        chunk_paths = save_huggingface_dataset_chunks_streamed(
            json_dir=json_dir,
            image_dir=image_dir,
            code_length=code_length,
            output_dir=save_dir
        )
        if not chunk_paths:
            raise SystemExit("No chunks were created; check paths and filters.")
        full_dataset = load_saved_chunks(chunk_paths)
    num_shards = 10
    for idx in range(num_shards):
        shard = full_dataset.shard(num_shards=num_shards, index=idx)
        shard_path = os.path.join(save_dir, f"shard_{idx:02d}")
        shard.save_to_disk(shard_path)