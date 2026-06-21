import json
import os

from utils import clean_tikz_code


def gather_data(folder_name, folder_path):
    idx = 0
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".tex"):
            with open(file_path, "r") as f:
                tikz_raw = f.read().strip()
                tikz_cleaned = clean_tikz_code(tikz_raw)
                title = os.path.splitext(file_name)[0]
                data_to_save = {
                    "title": title,
                    "description": title,
                    "code": tikz_cleaned,
                    "file_id": f"{folder_name}_{idx}"
                }
                all_data.append(data_to_save)
                idx += 1
    return all_data


if __name__ == "__main__":
    folder_name = "tikz_drawings"
    folder_path = os.path.abspath("datikz/raw/tikz_drawings")
    data = gather_data(folder_name, folder_path)
    with open(f"datikz/data/{folder_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)