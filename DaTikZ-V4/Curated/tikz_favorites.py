import json
import os

from utils import clean_tikz_code


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if file_name.endswith(".tex"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    tikz_raw = f.read().strip()
                    tikz_cleaned = clean_tikz_code(tikz_raw)
                    filename = os.path.basename(file_path).split(".")[0].replace("_", " ")
                    descr, _, tags = filename.partition("+")
                    title = f"{descr} ({tags.replace('+', ', ')})" if tags else descr
                    all_data.append({
                        "title": title,
                        "description": title,
                        "code": tikz_cleaned,
                        "file_id": f"{repo_name}_{idx}"
                    })
                    idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "tikz_favorites"
    repo_path = os.path.abspath("datikz/repos/" + repo_name)
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)