import json
import os
import re

from TexSoup import TexSoup
from utils import clean_tikz_code, is_non_visual_or_uncompilable_tikz


class CodeExample:
    def __init__(self, code, hidden=False, code_only=False, pre=None, post=None, preamble=None, render_instead=None):
        preamble = (preamble + "\n") if preamble else ""
        pre = (pre + "\n") if pre else ""
        code = render_instead if render_instead else code
        post = ("\n" + post) if post else ""
        self.code = self.format_as_document(preamble, pre + code + post)
        self.visible = not (hidden or code_only)

    def format_as_document(self, preamble, code):
        cls = "\\documentclass[tikz]{standalone}\n"
        doc = "{cls}{preamble}\n\\begin{{document}}\n\n{code}\n\n\\end{{document}}"
        return doc.format(cls=cls, preamble=preamble, code=code)


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
        soup = TexSoup(example, tolerance=1)
        code = soup.codeexample.expr.string.strip()
        args = dict()
        soupiter = iter(soup.codeexample.args[0].all)
        while item := next(soupiter, None):
            assert isinstance(item, str)
            for t in item.split(","):
                match t.strip():
                    case ("hidden" | "code only") as arg:
                        args[arg.replace(" ", "_")] = True
                    case ('pre=' | 'post=' | 'preamble=' | 'render instead=') as arg:
                        args[arg.replace(" ", "_")[:-1]] = next(soupiter).string.strip()
                    case arg if arg.endswith("="):
                        next(soupiter)

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


def gather_data(repo_name, repo_path):
    idx = 0
    all_data = []
    for root, _, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith(".tex"):
                file_path = os.path.join(root, filename)
                title = os.path.splitext(filename)[0].split("-")[-1]
                with open(file_path, "r") as f:
                    tex_content = f.read().strip()
                    for example, section_title, description in extract_examples(tex_content):
                        total_title = f"{title}: {section_title}" if section_title else title
                        if example.visible:
                            tikz_cleaned = clean_tikz_code(example.code)
                            non_compilable_visual = is_non_visual_or_uncompilable_tikz(tikz_cleaned)
                            if non_compilable_visual:
                                continue
                            all_data.append({
                                "title": total_title.strip(),
                                "description": description.strip(),
                                "code": tikz_cleaned,
                                "file_id": f"{repo_name}_{idx}"
                            })
                            idx += 1
    return all_data


if __name__ == "__main__":
    repo_name = "pgf_manual"
    repo_path = os.path.abspath("datikz/repos/pgf/doc/generic/pgf")
    data = gather_data(repo_name, repo_path)
    with open(f"datikz/data/{repo_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)