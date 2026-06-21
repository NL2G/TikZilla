import json
import re
from string import punctuation
from urllib.parse import urljoin as join
from urllib.request import urlopen

from bs4 import BeautifulSoup


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


def clean_tikz_code(tikz_raw):
    tikz_lines = tikz_raw.splitlines()
    return "\n".join(
        line for line in tikz_lines if not line.lstrip().startswith('%')
    ).strip()


def extract_tikz_blocks(tikz_code):
    preamble_match = re.split(r'\\begin{document}', tikz_code, maxsplit=1, flags=re.IGNORECASE)
    if len(preamble_match) != 2:
        return []
    preamble, rest = preamble_match
    tikz_blocks = re.findall(
        r'(\\begin{tikzpicture}.*?\\end{tikzpicture})',
        rest,
        flags=re.DOTALL
    )
    full_docs = [f"{preamble}\n\\begin{{document}}\n{tikz}\n\\end{{document}}" for tikz in tikz_blocks]
    return full_docs


def gather_data(repo_name, url):
    idx = 0
    all_data = []
    example_list = join(url, '/list')
    soup = BeautifulSoup(urlopen(example_list), 'html.parser')
    content = soup.find(id="upl-list-1493")
    for link in content.find_all("a"):
        uri = join(url, link.get("href"))
        example = BeautifulSoup(urlopen(uri), "html.parser")
        content = example.find('div', attrs={"class": "entry-content"})
        full_title = example.title.text.strip()
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
    repo_name = "texample_net"
    url = 'https://texample.net'
    data = gather_data(repo_name, url)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)