import json
import os
import re

from utils import clean_tikz_code


def extract_tikzpictures(code):
    pattern = re.compile(r'(\\begin{tikzpicture}.*?\\end{tikzpicture})', re.DOTALL)
    return pattern.findall(code)


def split_preamble_and_tikz(content):
    tikz_matches = extract_tikzpictures(content)
    if not tikz_matches:
        return None, []

    first_tikz_index = content.find(tikz_matches[0])
    preamble = content[:first_tikz_index].strip()
    tikz_blocks = tikz_matches
    return preamble, tikz_blocks


def standardize_tikz_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        raw_cleaned = clean_tikz_code(raw)
        preamble, tikz_blocks = split_preamble_and_tikz(raw_cleaned)

        documents = []
        for i, tikz in enumerate(tikz_blocks):
            document = (
                "\\documentclass[tikz]{standalone}\n\n"
                + (preamble + "\n\n" if preamble else "")
                + "\\begin{document}\n"
                + tikz.strip() + "\n"
                + "\\end{document}"
            )
            documents.append(document)

        return documents


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for file_name in os.listdir(repo_path):
        if not file_name.endswith(".pgf"):
            continue

        file_path = os.path.join(repo_path, file_name)
        title_base = os.path.splitext(file_name)[0]
        tikz_docs = standardize_tikz_documents(file_path)

        for i, code in enumerate(tikz_docs):
            entry = {
                "title": f"{title_base}_{i+1}" if len(tikz_docs) > 1 else title_base,
                "description": f"{title_base}_{i+1}" if len(tikz_docs) > 1 else title_base,
                "code": code,
                "file_id": f"{repo_name}_{idx}"
            }
            all_data.append(entry)
            idx += 1

    return all_data


if __name__ == "__main__":
    repo_name = "walmes_tikz"
    repo_path = os.path.abspath("datikz/repos/Tikz/src")
    data = gather_data(repo_name, repo_path)
    os.makedirs("datikz/data", exist_ok=True)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)