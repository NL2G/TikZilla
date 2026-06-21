import json
import os

from yaml import Loader, load as yload
from utils import clean_tikz_code


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for folder_name in os.listdir(repo_path):
        folder_path = os.path.join(repo_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".tex"):
                    with open(file_path, "r") as f, open(file_path.replace(".tex", ".yml")) as g:
                        code_raw = f.read().strip()
                        code_cleaned = clean_tikz_code(code_raw)
                        yaml = yload(g.read(), Loader)
                        title = yaml.get("title")
                        description = yaml.get("description")
                        data_to_save = {
                            "title": title,
                            "description": description,
                            "code": code_cleaned,
                            "file_id": f"{repo_name}_{idx}"
                        }
                        idx += 1
                        all_data.append(data_to_save)
    return all_data


if __name__ == "__main__":
    repo_name = "janosh_tikz"
    repo_path = os.path.abspath("datikz/repos/diagrams/assets")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)