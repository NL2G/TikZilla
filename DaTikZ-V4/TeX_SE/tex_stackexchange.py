import json
import os
import re

from collections import defaultdict
from lxml import etree
from bs4 import BeautifulSoup
from utils import clean_tikz_code


USEPACKAGES = {
    "circuitikz": [r"\\begin{circuitikz}", r"\\draw\[[^\]]*circuit[^\]]*\]"],
    "pgfplots": [r"\\begin{axis}", r"\\begin{polaraxis}", r"\\addplot"],
    "tikzcd": [r"\\begin{tikzcd}"],
    "forest": [r"\\begin{forest}"],
    "smartdiagram": [r"\\smartdiagram"],
    "mindmap": [r"\\smartdiagram", r"mindmap", r"\\usetikzlibrary\{mindmap\}"],
    "graphicx": [r"\\includegraphics"],
    "positioning": [r"\\usepackage\{positioning\}"],
}


TIKZLIBRARIES = {
    "arrows.meta": [r"\barrow", r"arrows.meta", r"stealth"],
    "decorations.pathmorphing": [r"snake", r"decorate", r"decoration"],
    "positioning": [r"\\node.*(above|below|left|right|of)="],
    "shapes": [r"\\node.*(rectangle|circle|diamond)"],
    "calc": [r"\$.*?\+.*?\$"],
    "fit": [r"fit=", r"node\[fit="],
    "backgrounds": [r"background", r"layer"],
    "graphs": [r"\\graph", r"graph\[.*\]"],
    "matrix": [r"matrix of nodes"],
    "mindmap": [r"mindmap", r"concept"],
}


def is_question(attribs):
    return attribs.get("PostTypeId") == "1"


def is_answer(attribs):
    return attribs.get("PostTypeId") == "2"


def is_accepted_answer(answer, question):
    return question.get("AcceptedAnswerId") == answer.get("Id")


def has_answers(question):
    return bool(question.get("AnswerCount"))


def trim_attribs(attribs, attrib_type="question"):
    if attrib_type == "question":
        keep = ['Id', 'Body', 'Title', 'Tags', 'AnswerCount', 'AcceptedAnswerId']
        attribs = {k: attribs[k] for k in keep if k in attribs}
        attribs["ParsedAnswers"] = 0
        attribs["Answers"] = {}
        return attribs
    elif attrib_type == "answer":
        keep = ['Id', 'Body', 'Score', 'CreationDate', 'ParentId']
        return {k: attribs[k] for k in keep if k in attribs}
    else:
        raise ValueError("Unknown attribute type")


def detect_required_packages(code):
    packages = set(["tikz"])
    libraries = set()
    for pkg, patterns in USEPACKAGES.items():
        for pat in patterns:
            if re.search(pat, code):
                packages.add(pkg)
                break
    for lib, patterns in TIKZLIBRARIES.items():
        for pat in patterns:
            if re.search(pat, code):
                libraries.add(lib)
                break
    return packages, libraries


def build_preamble(packages, libraries):
    preamble = "\\documentclass[tikz]{standalone}\n"
    for pkg in sorted(packages):
        preamble += f"\\usepackage{{{pkg}}}\n"
    if libraries:
        preamble += f"\\usetikzlibrary{{{', '.join(sorted(libraries))}}}\n"
    return preamble


def standardize_documentclass(code):
    code = re.sub(r"\\documentclass(\[.*?\])?\{(article|report|book|scrartcl)\}", r"\\documentclass[tikz]{standalone}", code)
    return code


def filter_tikz_code(code):
    lines = code.splitlines()
    filtered = []
    input_pattern = re.compile(r'\\input\{[^}]*\}')
    includegraphics_pattern = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}')
    includepdf_pattern = re.compile(r'\\includepdf(?:\[[^\]]*\])?\{[^}]*\}')
    for line in lines:
        cleaned = input_pattern.sub('', line)
        cleaned = includegraphics_pattern.sub('', cleaned)
        cleaned = includepdf_pattern.sub('', cleaned)
        cleaned = re.sub(r'\{\s*\}', '{}', cleaned)
        if any(kw in cleaned for kw in ['\\maketitle', '\\tableofcontents']):
            continue
        if cleaned.strip() and not cleaned.strip().startswith('%'):
            filtered.append(cleaned)
    return '\n'.join(filtered)


def extract_code_blocks(soup):
    tikz_envs = [
        "tikzpicture", "forest", "circuitikz", "pgfpicture", "axis", "scope", "tikzcd",
        "polaraxis", "semilogyaxis", "loglogaxis", "mindmap", "smartdiagram", "graphdrawing"
    ]
    for pre in soup.find_all("pre"):
        code = getattr(pre.code, "text", None)
        if not code:
            continue
        if not any(f"\\begin{{{env}}}" in code and f"\\end{{{env}}}" in code for env in tikz_envs):
            continue
        needs_doc = r"\begin{document}" not in code
        needs_class = r"\documentclass" not in code
        if needs_class or needs_doc:
            packages, libraries = detect_required_packages(code)
            preamble = build_preamble(packages, libraries)
            final_code = preamble
            final_code += "\n\\begin{document}\n"
            final_code += code.strip()
            final_code += "\n\\end{document}"
        else:
            final_code = code
        cleaned_final_code = clean_tikz_code(final_code)
        standardized_code = standardize_documentclass(cleaned_final_code)
        filtered_cleaned_final_code = filter_tikz_code(standardized_code)
        yield filtered_cleaned_final_code


class StackexchangeParser:
    def __init__(self, xml_path, base_name, tags=None, min_score=-10000):
        self.xml_path = xml_path
        self.base_name = base_name
        self.questions = defaultdict(lambda: None)
        self.tags = [] if tags is None else tags
        self.min_score = min_score

    def load(self):
        for _, elem in etree.iterparse(self.xml_path, events=("end",), recover=True):
            if elem.tag != "row":
                continue
            attribs = defaultdict(lambda: None, elem.attrib)
            if is_question(attribs):
                if has_answers(attribs) and all(f"<{tag}>" in attribs.get("Tags", "") for tag in self.tags):
                    self.questions[attribs["Id"]] = trim_attribs(attribs, "question")
            elif is_answer(attribs):
                self._add_answer(attribs)
                yield from self._check_complete(attribs)
            elem.clear()

    def _is_valid_answer(self, answer):
        return int(answer.get("Score", 0)) >= self.min_score

    def _add_answer(self, answer):
        parent_id = answer.get("ParentId")
        if not parent_id or not self.questions[parent_id]:
            return
        parent = self.questions[parent_id]
        if is_accepted_answer(answer, parent) or self._is_valid_answer(answer):
            parent["Answers"][answer["Id"]] = trim_attribs(answer, "answer")
        parent["ParsedAnswers"] += 1

    def _check_complete(self, answer):
        parent_id = answer.get("ParentId")
        parent = self.questions.get(parent_id)
        if not parent:
            return
        if int(parent.get("ParsedAnswers", 0)) == int(parent.get("AnswerCount", 0)):
            self.questions.pop(parent_id, None)
            if not parent["Answers"]:
                return
            title = parent.get("Title", "")
            # soup = BeautifulSoup(parent.get("Body", ""), "html.parser")
            body = parent.get("Body")
            if not body:
                return
            soup = BeautifulSoup(body, "html.parser")
            context_blocks = list(extract_code_blocks(soup))
            for pre in soup.find_all("pre"):
                pre.decompose()
            description = soup.text.strip()
            answers_out = []
            for a in parent["Answers"].values():
                soup_a = BeautifulSoup(a["Body"], "html.parser")
                answer_code = list(extract_code_blocks(soup_a))
                if not answer_code:
                    continue

                answers_out.append({
                    "code": answer_code,
                    "file_id": [f"{self.base_name}_{a['Id']}_{i}" for i in range(len(answer_code))]
                })
            if not answers_out:
                return
            for i, question_code in enumerate(context_blocks):
                yield {
                    "description": "\n\n".join([title, description]).strip(),
                    "code": question_code,
                    "file_id": f"{self.base_name}_{parent.get('Id')}_{i}"
                }
            for a in answers_out:
                for answer_code, answer_id in zip(a["code"], a["file_id"]):
                    yield {
                        "description": "\n\n".join([title, description]).strip(),
                        "code": answer_code,
                        "file_id": answer_id
                    }


def write_chunked(generator, output_dir, base_name, chunk_size):
    os.makedirs(output_dir, exist_ok=True)
    chunk = []
    idx = 0
    for item in generator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            with open(os.path.join(output_dir, f"{base_name}_{idx}.json"), "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=4, ensure_ascii=False)
            idx += 1
            chunk = []
    if chunk:
        with open(os.path.join(output_dir, f"{base_name}_{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    base_name = "math" # or tex or math
    xml_file = f"datikz/raw/{base_name}.stackexchange.com/Posts.xml"
    output_dir = "datikz/data"
    chunk_size = 5000
    parser = StackexchangeParser(xml_file, base_name)
    write_chunked(parser.load(), output_dir, base_name, chunk_size)