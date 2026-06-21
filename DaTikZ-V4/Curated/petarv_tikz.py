import json
import os

from bs4 import BeautifulSoup
from markdown import markdown
from utils import clean_tikz_code


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for folder_name in os.listdir(repo_path):
        folder_path = os.path.join(repo_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        tex_code = None
        title = None
        description = None
        for root, _, files in os.walk(folder_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filename.endswith(".tex") and tex_code is None:
                    with open(filepath, "r", encoding="utf-8") as tex_file:
                        tikz_raw = tex_file.read().strip()
                        tex_code = clean_tikz_code(tikz_raw)
                elif filename.endswith(".md") and title is None:
                    with open(filepath, "r", encoding="utf-8") as readme_file:
                        soup = BeautifulSoup(markdown(readme_file.read()), "html.parser")
                        h1 = soup.find("h1")
                        title = h1.text if h1 else "Untitled"
                        notes_header = soup.find(lambda tag: tag.name == "h2" and "Notes" in tag.text)
                        if notes_header:
                            next_p = notes_header.find_next_sibling("p")
                            description = next_p.text if next_p else title
                        else:
                            description = title
        if tex_code and title:
            all_data.append({
                "title": title,
                "description": description,
                "code": tex_code,
                "file_id": f"{repo_name}_{idx}"
            })
            idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "petarv_tikz"
    repo_path = os.path.abspath("datikz/repos/TikZ")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)