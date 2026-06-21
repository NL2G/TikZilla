import json
import os

from utils import clean_tikz_code


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for folder_name in os.listdir(repo_path):
        folder_path = os.path.join(repo_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        first_title = os.path.basename(folder_path)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.endswith(".tex"):
                    continue
                filepath = os.path.join(root, filename)
                second_title = os.path.splitext(filename)[0]
                filepath_keywords = os.path.join(root, f"{second_title}_keywords.txt")
                filepath_description = os.path.join(root, f"{second_title}_description.txt")
                with open(filepath, "r", encoding="utf-8") as tex_file:
                    tikz_raw = tex_file.read().strip()
                    tikz_cleaned = clean_tikz_code(tikz_raw)
                keywords = ""
                description = ""
                if os.path.isfile(filepath_keywords):
                    with open(filepath_keywords, "r", encoding="utf-8") as key_file:
                        keywords = key_file.read().strip()
                if os.path.isfile(filepath_description):
                    with open(filepath_description, "r", encoding="utf-8") as desc_file:
                        description = desc_file.read().strip()
                all_data.append({
                    "title": f"{first_title}: {second_title}",
                    "description": f"{description} (Keywords: {keywords})",
                    "code": tikz_cleaned,
                    "file_id": f"{repo_name}_{idx}"
                })
                idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "latex_diagrams"
    repo_path = os.path.abspath("datikz/repos/LatexDiagrams")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)