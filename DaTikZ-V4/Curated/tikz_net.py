import json
import re

from itertools import count
from string import punctuation
from urllib.request import urlopen
from bs4 import BeautifulSoup
from utils import clean_tikz_code


def find_description_block(pre_tag):
    desc_parts = []
    for sibling in pre_tag.previous_siblings:
        if sibling.name == "pre":
            break
        if sibling.name == "p":
            text = sibling.get_text(strip=True)
            if text:
                desc_parts.insert(0, text)
        elif isinstance(sibling, str) and sibling.strip():
            desc_parts.insert(0, sibling.strip())
        elif sibling.name and sibling.name.startswith("h"):
            break
    return " ".join(desc_parts).strip()


def find_nearest_heading(pre_tag):
    for sibling in pre_tag.previous_siblings:
        if getattr(sibling, "name", "").startswith("h"):
            return sibling.get_text(strip=True)
    return None


def clean_description(desc, title):
    phrases_to_remove = [
        "Edit and compile if you like:",
        "Edit and compile the code if you like.",
        "Full code to edit and compile if you like:",
        "Edit and compile the full code below if you like."
    ]
    for phrase in phrases_to_remove:
        desc = desc.replace(phrase, "")
    desc = desc.strip()
    if not desc:
        return "No Description"
    if title.lower() not in desc.lower():
        desc = f"{title}. {desc}"
        if desc[-1] not in punctuation:
            desc += "."
    return " ".join(desc.split())


def extract_tikz_blocks(tikz_code):
    preamble_match = re.split(r'\\begin{document}', tikz_code, maxsplit=1, flags=re.IGNORECASE)
    if len(preamble_match) != 2:
        return []
    preamble, rest = preamble_match
    tikz_library_keywords = {
        'circuit': 'circuits',
        'mindmap': 'mindmap',
        'shapes': 'shapes',
        'arrows': 'arrows',
        'positioning': 'positioning',
        'automata': 'automata',
        'backgrounds': 'backgrounds',
        'fit': 'fit',
        'matrix': 'matrix',
        'patterns': 'patterns',
        'trees': 'trees',
        '3d': '3d'
    }
    if 'tdplot_main_coords' in tikz_code or 'tdplot' in tikz_code:
        if r'\usepackage{tikz-3dplot}' not in preamble:
            preamble += '\n\\usepackage{tikz-3dplot}'
    detected_libraries = set()
    for keyword, library in tikz_library_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', tikz_code):
            detected_libraries.add(library)
    if detected_libraries:
        libraries_line = f"\\usetikzlibrary{{{','.join(sorted(detected_libraries))}}}"
        if '\\usetikzlibrary' not in preamble:
            preamble += f"\n{libraries_line}"
    tikz_blocks = re.findall(
        r'(\\begin{tikzpicture}.*?\\end{tikzpicture})',
        rest,
        flags=re.DOTALL
    )
    full_docs = [f"{preamble}\n\\begin{{document}}\n{tikz}\n\\end{{document}}" for tikz in tikz_blocks]
    return full_docs


def gather_data(repo_name, base_url):
    idx = 0
    all_data = []
    for page in count(0):
        example_list = f"{base_url}?infinity=scrolling&action=infinite_scroll&page={page}"
        json_example = json.load(urlopen(example_list))
        if json_example['type'] == "empty":
            break
        for article in BeautifulSoup(json_example['html'], 'html.parser').find_all('article'):
            soup = BeautifulSoup(urlopen(article.a.get('href')), 'html.parser')
            content = soup.find('div', attrs={"class": "entry-content"})
            full_title = soup.title.text.strip()
            title = full_title.split("–")[0].strip()
            if content:
                for pre in content.find_all("pre"):
                    tikz_raw = pre.text
                    tikz_clean = clean_tikz_code(tikz_raw)
                    if not tikz_clean:
                        continue
                    desc_raw = find_description_block(pre)
                    desc = clean_description(desc_raw, title)
                    tikz_docs = extract_tikz_blocks(tikz_clean)
                    for tikz_doc in tikz_docs:
                        all_data.append({
                            "title": title,
                            "description": desc,
                            "code": tikz_doc,
                            "file_id": f"{repo_name}_{idx}"
                        })
                        idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "tikz_net"
    repo_path = "https://tikz.net"
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)