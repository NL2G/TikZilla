import json
import os
import re

from utils import clean_tikz_code


def clean_and_inject_packages(code):
    code = re.sub(r'\\usepackage\{.*?tikz[-]?network.*?\}', '', code)
    if r'\documentclass' not in code:
        code = "\\documentclass{standalone}\n" + code
    if r'\usepackage{tikz-network}' not in code:
        code = code.replace(r'\documentclass{standalone}', '\\documentclass{standalone}\n\\usepackage{tikz-network}')
    return code.strip()


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for folder in os.listdir(repo_path):
        folder_path = os.path.join(repo_path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".tex"):
                with open(file_path, "r", encoding="utf-8") as f:
                    tikz_raw = f.read().strip()
                    tikz_cleaned = clean_tikz_code(tikz_raw)
                    tikz_cleaned = clean_and_inject_packages(tikz_cleaned)
                    if not tikz_cleaned:
                        continue
                    title = os.path.splitext(file_name)[0]
                    data_to_save = {
                        "title": title,
                        "description": title,
                        "code": tikz_cleaned,
                        "file_id": f"{repo_name}_{idx}"
                    }
                    all_data.append(data_to_save)
                    idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "tikz_network"
    repo_path = os.path.abspath("datikz/repos/tikz-network/examples")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)