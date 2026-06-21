import os
import json
import hashlib

from glob import glob
from tqdm import tqdm


reference_dir = "ArXiv/data"
target_dir = "ArXiv/data_raw"


def hash_code(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    reference_hashes = set()
    for ref_file in tqdm(glob(os.path.join(reference_dir, "all_*.json"))):
        try:
            with open(ref_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    reference_hashes.add(hash_code(item["code"]))
        except Exception:
            continue
    for target_file in tqdm(glob(os.path.join(target_dir, "arxiv_src_*.json"))):
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        local_seen_hashes = set()
        deduped_data = []
        for item in data:
            code_hash = hash_code(item["code"])
            if code_hash in reference_hashes or code_hash in local_seen_hashes:
                continue
            local_seen_hashes.add(code_hash)
            deduped_data.append(item)
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(deduped_data, f, indent=2, ensure_ascii=False)
        except Exception:
            continue