import requests
import json
import re

from bs4 import BeautifulSoup
from utils import clean_tikz_code


def clean_title(title):
    cleaned = re.sub(r"^Figure\s[\d.,\s]+–\s*", "", title)
    return cleaned.strip()


def gather_data(repo_name, urls):
    idx = 0
    extracted_data = []
    for url in urls:
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        example_blocks = soup.find_all('pre')
        for example in example_blocks:
            title_tag = example.find_previous('p', style="text-align: center;")
            title = clean_title(title_tag.get_text(strip=True)) if title_tag else "Untitled"
            code = example.get_text(strip=True)
            cleaned_code = clean_tikz_code(code)
            data = {
                "title": title,
                "description": title,
                "code": cleaned_code,
                "file_id": f"{repo_name}_{idx}"
            }
            extracted_data.append(data)
            idx += 1
    return extracted_data


if __name__ == "__main__":
    repo_name = "tikz_org"
    urls = [
    "https://tikz.org/examples/chapter-01-getting-started-with-tikz/",
    "https://tikz.org/examples/chapter-02-creating-the-first-tikz-images/",
    "https://tikz.org/examples/chapter-03-drawing-positioning-and-aligning-nodes/",
    "https://tikz.org/examples/chapter-04-drawing-edges-and-arrows/",
    "https://tikz.org/examples/chapter-05-using-styles-and-pics/",
    "https://tikz.org/examples/chapter-06-drawing-trees-and-graphs/",
    "https://tikz.org/examples/chapter-07-filling-and-clipping/",
    "https://tikz.org/examples/chapter-08-decorating-paths/",
    "https://tikz.org/examples/chapter-09-using-layers-and-transparency/",
    "https://tikz.org/examples/chapter-10-calculating-with-coordinates-and-paths/",
    "https://tikz.org/examples/chapter-11-transforming-coordinates-and-canvas/",
    "https://tikz.org/examples/chapter-12-drawing-smooth-curves/",
    "https://tikz.org/examples/chapter-13-plotting-in-2d-and-3d/",
    "https://tikz.org/examples/chapter-14-drawing-diagrams/",
    "https://tikz.org/examples/chapter-15-having-fun-with-tikz/"
    ]
    data = gather_data(repo_name, urls)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)