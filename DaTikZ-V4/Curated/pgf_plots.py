import json
import os
import re

from TexSoup import TexSoup
from utils import clean_tikz_code


DEFAULT_PREAMBLE = r"""
\usepackage{pgfplots}
\pgfplotsset{compat=1.18} % Adjust to match your TeX version
\usepgfplotslibrary{groupplots, dateplot, statistics}
\usepackage{pgfplotstable}
\usetikzlibrary{positioning, calc, arrows.meta, decorations.pathreplacing, shapes.multipart}
"""

class CodeExample:
    def __init__(self, code, hidden=False, code_only=False, pre=None, post=None, preamble=None, render_instead=None):
        preamble = (preamble + "\n") if preamble else detect_preamble(code)
        pre = (pre + "\n") if pre else ""
        code = render_instead if render_instead else code
        post = ("\n" + post) if post else ""
        self.code = self.format_as_document(preamble, pre + code + post)
        self.visible = not (hidden or code_only)

    def format_as_document(self, preamble, code):
        cls = "\\documentclass[tikz]{standalone}\n"
        doc = "{cls}{preamble}\n\\begin{{document}}\n\n{code}\n\n\\end{{document}}"
        return doc.format(cls=cls, preamble=preamble, code=code)


def detect_preamble(code):
    lines = [r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    if "groupplot" in code:
        lines.append(r"\usepgfplotslibrary{groupplots}")
    if "dateplot" in code:
        lines.append(r"\usepgfplotslibrary{dateplot}")
    if "pgfplotstabletypeset" in code:
        lines.append(r"\usepackage{pgfplotstable}")
    if any(k in code for k in ("node", "positioning", "calc", "arrows.meta", "shapes.multipart")):
        lines.append(r"\usetikzlibrary{positioning,calc,arrows.meta,shapes.multipart}")
    return "\n".join(lines)


def lines_startwith(string, prefix):
    return all(line.startswith(prefix) for line in string.splitlines())


def lines_removeprefix(string, prefix):
    return "".join(line.removeprefix(prefix) for line in string.splitlines(keepends=True))


def extract_examples(doc):
    example_regex = r"(([^\n]*?)\\begin{codeexample}.*?\\end{codeexample})"
    section_regex = r"(\\(?:subsubsection|subsection|section)\*?{.*?})"
    matches = list(re.finditer(example_regex, doc, re.DOTALL))
    last_end = 0
    for match in matches:
        full_block, prefix = match.groups()
        start_idx, end_idx = match.span()
        if len(prefix) > 0 and not prefix.strip():
            example = full_block.lstrip()
        elif lines_startwith(dedented := full_block.lstrip(), "%"):
            example = lines_removeprefix(dedented, "%")
        else:
            example = full_block.removeprefix(prefix)
        try:
            soup = TexSoup(example, tolerance=1)
            code = soup.codeexample.expr.string.strip()
            args = dict()
            try:
                soup_args = list(soup.codeexample.args[0].all)
            except:
                soup_args = list(soup.codeexample.args.all)
            i = 0
            while i < len(soup_args):
                item = soup_args[i]
                if isinstance(item, str):
                    for t in item.split(","):
                        t = t.strip()
                        match t:
                            case ("hidden" | "code only") as arg:
                                args[arg.replace(" ", "_")] = True
                            case ('pre=' | 'post=' | 'preamble=' | 'render instead=') as arg:
                                i += 1
                                if i < len(soup_args):
                                    args[arg.replace(" ", "_")[:-1]] = soup_args[i].string.strip()
                            case _ if t.endswith("="):
                                i += 1
                i += 1
            pre_block = doc[last_end:start_idx]
            last_end = end_idx
            section_match = list(re.finditer(section_regex, pre_block))[-1] if re.search(section_regex, pre_block) else None
            section_text = ""
            pre_desc_text = ""
            if section_match:
                _, sec_end = section_match.span()
                section_line = section_match.group(1)
                section_text = re.search(r"{(.*?)}", section_line).group(1).strip()
                pre_desc_text = pre_block[sec_end:].strip()
                pre_desc_text = re.sub(r"\s+", " ", pre_desc_text)
                pre_desc_text = pre_desc_text[:500]
            yield CodeExample(code=code, **args), section_text, pre_desc_text
        except EOFError:
            continue


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for root, _, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith(".tex"):
                file_path = os.path.join(root, filename)
                title = os.path.splitext(filename)[0].split(".")[-1]
                with open(file_path, "r") as f:
                    tex_content = f.read().strip()
                    for example, section_title, description in extract_examples(tex_content):
                        total_title = f"{title}: {section_title}" if section_title else title
                        if example.visible:
                            tikz_cleaned = clean_tikz_code(example.code)
                            all_data.append({
                                "title": total_title.strip(),
                                "description": description.strip(),
                                "code": tikz_cleaned,
                                "file_id": f"{repo_name}_{idx}"
                            })
                            idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "pgf_plots"
    repo_path = os.path.abspath("datikz/repos/pgfplots/doc/latex/pgfplots")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)