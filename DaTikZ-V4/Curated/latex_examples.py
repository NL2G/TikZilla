import json
import os

from utils import clean_tikz_code


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for folder_name in os.listdir(repo_path):
        folder_path = os.path.join(repo_path, folder_name)
        for _, _, files in os.walk(folder_path):
            for filename in files:
                filepath = os.path.join(folder_path, filename)
                if filename.endswith(".tex"):
                    with open(filepath, "r") as tex_file:
                        tikz_raw = tex_file.read().strip()
                        tikz_cleaned = clean_tikz_code(tikz_raw)
                        title = os.path.basename(filepath).removesuffix(".tex").replace("-", " ")
                        all_data.append({
                            "title": title,
                            "description": title,
                            "code": tikz_cleaned,
                            "file_id": f"{repo_name}_{idx}"
                        })
                        idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "latex_examples"
    repo_path = os.path.abspath("datikz/repos/LaTeX-examples/tikz")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)