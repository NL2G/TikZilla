import json
import os
import zipfile

from pathlib import Path
from PIL import Image

SPLITS_TO_USE = ("test-A", "test-B", "test-C") #train, val, test-A, test-B, test-C
OUTPUT_DIR_NAME = "spiqa"
IMAGES_SUBDIR_NAME = "images"
OUTPUT_JSON_NAME = "spiqa.json"


def find_spiqa_root(start_dir):
    for path in start_dir.rglob("SPIQA_val.json"):
        if path.name == "SPIQA_val.json":
            train_val_dir = path.parent
            root = train_val_dir.parent
            return root
    raise FileNotFoundError(
        "Could not find SPIQA_val.json under "
        f"{start_dir}. Please ensure you've downloaded the dataset with "
        "snapshot_download(repo_id='google/spiqa', local_dir='.') and that "
        "this script is placed somewhere inside or above that folder."
    )


def build_zip_name_map(zf):
    name_map = {}
    for info in zf.infolist():
        base = os.path.basename(info.filename)
        if not base:
            continue
        name_map[base] = info.filename
    return name_map


def iter_spiqa_qa_items(paper_obj):
    qa_list = paper_obj.get("qa", [])
    for qa in qa_list:
        ref = (
            qa.get("reference")
            or qa.get("reference_figure")
            or qa.get("reference_figures")
        )
        if not ref:
            continue
        if isinstance(ref, str):
            ref_list = [ref]
        elif isinstance(ref, list):
            ref_list = [x for x in ref if isinstance(x, str)]
        else:
            continue
        rationale = (
            qa.get("explanation")
            or qa.get("rationale")
            or qa.get("image_description")
        )
        if not rationale:
            continue
        rationale_clean = " ".join(str(rationale).split())
        for ref_item in ref_list:
            yield ref_item, rationale_clean


def process_split(split_name, base_dir, images_dir):
    records = []
    if split_name == "train":
        json_path = base_dir / "train_val" / "SPIQA_train.json"
        zip_path = base_dir / "train_val" / "SPIQA_train_val_Images.zip"
    elif split_name == "val":
        json_path = base_dir / "train_val" / "SPIQA_val.json"
        zip_path = base_dir / "train_val" / "SPIQA_train_val_Images.zip"
    elif split_name == "test-A":
        json_path = base_dir / "test-A" / "SPIQA_testA.json"
        zip_path = base_dir / "test-A" / "SPIQA_testA_Images.zip"
    elif split_name == "test-B":
        json_path = base_dir / "test-B" / "SPIQA_testB.json"
        zip_path = base_dir / "test-B" / "SPIQA_testB_Images.zip"
    elif split_name == "test-C":
        json_path = base_dir / "test-C" / "SPIQA_testC.json"
        zip_path = base_dir / "test-C" / "SPIQA_testC_Images.zip"
    else:
        return records
    if not json_path.exists():
        return records
    if not zip_path.exists():
        zf = None
        name_map = {}
    else:
        zf = zipfile.ZipFile(zip_path, "r")
        name_map = build_zip_name_map(zf)
    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    seen_images = set()
    paper_ids = list(metadata.keys())
    for _, paper_obj in metadata.items():
        for ref_figure, rationale in iter_spiqa_qa_items(paper_obj):
            ref_filename = os.path.basename(ref_figure)
            figure_id = os.path.splitext(ref_filename)[0]
            if "table" in figure_id.lower():
                continue
            records.append(
                {
                    "figure_id": figure_id,
                    "new_caption": rationale,
                }
            )
            if zf is not None and ref_filename not in seen_images:
                inner_path = name_map.get(ref_filename)
                if inner_path is None:
                    continue
                target_path = images_dir / ref_filename
                with zf.open(inner_path) as src:
                    img = Image.open(src).convert("RGB")
                    orig_w, orig_h = img.size
                    scale = min(448 / orig_w, 448 / orig_h)
                    new_w = int(round(orig_w * scale))
                    new_h = int(round(orig_h * scale))
                    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
                    canvas = Image.new("RGB", (448, 448), (255, 255, 255))
                    left = (448 - new_w) // 2
                    top = (448 - new_h) // 2
                    canvas.paste(img_resized, (left, top))
                    canvas.save(target_path, format="PNG")
                seen_images.add(ref_filename)
    if zf is not None:
        zf.close()
    return records


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = find_spiqa_root(script_dir / "spiqa_raw")
    output_dir = base_dir / OUTPUT_DIR_NAME
    images_dir = output_dir / IMAGES_SUBDIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for split_name in SPLITS_TO_USE:
        recs = process_split(split_name, base_dir, images_dir)
        all_records.extend(recs)
    unique = {}
    for r in all_records:
        key = (r["figure_id"], r["new_caption"])
        unique[key] = r
    all_records = list(unique.values())
    output_json_path = output_dir / OUTPUT_JSON_NAME
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)