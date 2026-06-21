import os
import re
import signal
import json

from tqdm import tqdm
from pathlib import Path
from glob import glob
from os.path import join
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from multiprocessing import TimeoutError, Pool, get_context
from collections import namedtuple
from textwrap import dedent
from subprocess import check_output, CalledProcessError
from concurrent.futures import TimeoutError
from re import DOTALL, MULTILINE, findall, finditer, escape, search, split, sub
from TexSoup import TexSoup
from datasets import load_dataset
from ArXiv.demacro import TexDemacro, Error as DemacroError
from utils import clean_tikz_code


EXTERNAL_DEPENDENCY_PATTERNS = [
    r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}',
    r'\\input(\[[^\]]*\])?\{[^}]*\}',
    r'\\includepdf(\[[^\]]*\])?\{[^}]*\}',
    r'\\csvreader(\[[^\]]*\])?\s*\{[^}]*\}',
    r'\\pgfplotstableread(\[[^\]]*\])?\s*\{[^}]*\}',
    r'\\addplot\s+table(\[[^\]]*\])?\s*\{[^}]*\}',
    r'\\addplot\s+graphics(\[[^\]]*\])?\s*\{[^}]*\}',
]

DEFAULT_PREAMBLE = r"""
\documentclass[tikz]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usetikzlibrary{arrows.meta, decorations.pathreplacing, positioning, shapes, calc}
"""

SUPPORTED_ENVIRONMENTS = ["tikzpicture", "circuitikz", "tikzcd"]

SAME_ENV = True
REM_EXT = True
OTHER_IMG = True
TEXT_MENTION = True

TIMEOUT = 60
MAX_TIKZ_LENGTH = 100000


Tikz = namedtuple("TikZ", ['code', 'caption', 'label'])


def list_texlive_classes():
    try:
        texmfdist = check_output(["kpsewhich", "--var-value=TEXMFDIST"])
        texmf_path = texmfdist.strip().decode()
        cls_path = os.path.join(texmf_path, "ls-R")
        with open(cls_path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip().removesuffix(".cls") for line in f if line.endswith(".cls\n")]
    except (CalledProcessError, FileNotFoundError, OSError):
        return []


def extract_preamble(tex):
    include = ["documentclass", "tikz", "tkz", "pgf"]
    packages = ["inputenc", "fontenc", "fontspec", "amsmath", "amssymb", "color"]
    exclude = [r"\new", r"\renew"]
    preamble, *_ = tex.partition(r"\begin{document}")
    try:
        soup = TexSoup(preamble)
        statements = map(str, soup.children)
    except Exception as e:
        statements = preamble.splitlines()
    imports, macros = [], []
    tex_classes = list_texlive_classes()
    for stmt in statements:
        line = stmt.strip()
        if line.startswith("%") or not line:
            continue
        if any(line.startswith(pat) for pat in exclude):
            continue
        if any(pat in stmt for pat in include) or (line.startswith(r"\usepackage") and any(p in stmt for p in packages)):
            if line.startswith(r"\documentclass") and not any(c in stmt for c in tex_classes):
                imports.append(r"\documentclass{article}")
            else:
                imports.append(stmt)
        else:
            macros.append(stmt)
    return "\n".join(imports).strip(), "\n".join(macros).strip()


def expand_macros(macros, content, expand=True):
    try:
        ts = TexDemacro(macros=macros)
        return ts.process(content) if expand else "\n\n".join(ts.find(content)).strip()
    except (DemacroError, RecursionError, TypeError) as e:
        return content if expand else ""


def extract_color_definitions(macros, tikz_code):
    # definecolor_regex = r'^\s*\\definecolor(?:\[\w+?\])?\{(\w+?)\}\{\w+?\}\{.+?\}'
    definecolor_regex = r'^\s*\\definecolor(?:\[\w+?\])?\{([a-zA-Z0-9_\-]+)\}\{\w+?\}\{.+?\}'
    matches = []
    for color in finditer(definecolor_regex, macros, MULTILINE):
        name, definition = color.group(1), color.group().strip()
        if search(rf"\\b{name}\\b", tikz_code):
            matches.append(definition)
    return "\n".join(matches).strip()


def extract_label(figure_code):
    match = re.search(r"\\label\{([^\}]+)\}", figure_code)
    return match.group(1).strip() if match else ""


def split_paragraphs(text):
    return [p.strip() for p in split(r'\n\s*\n|(?=\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{)|(?=\\begin\{)', text) if p.strip()]


def extract_text_mentions(tex, tikz_labels, min_len=10, max_len=1000, N=1, M=3):
    paragraphs = split_paragraphs(tex)
    ref_pattern = re.compile(r'\\(?:ref|cref|autoref)\{([^}]+)\}')
    mentions = defaultdict(set)
    for para in paragraphs:
        matches = ref_pattern.findall(para)
        if not matches:
            continue
        para_len = len(para)
        refs_in_para = {
            ref.strip()
            for match in matches
            for ref in match.split(',')
            if ref.strip() in tikz_labels
        }
        if not refs_in_para:
            continue
        if para_len < min_len or para_len > max_len:
            sentences = sent_tokenize(para)
            for i, sent in enumerate(sentences):
                local_matches = ref_pattern.findall(sent)
                local_refs = {
                    ref.strip()
                    for match in local_matches
                    for ref in match.split(',')
                    if ref.strip() in tikz_labels
                }
                if not local_refs:
                    continue
                context = ' '.join(sentences[max(0, i - N):min(len(sentences), i + 1 + M)])
                for ref in local_refs:
                    mentions[ref].add(context)
        else:
            for ref in refs_in_para:
                mentions[ref].add(para)
    return {ref: list(texts) for ref, texts in mentions.items()}


def extract_caption(figure):
    try:
        if "\\caption{" not in figure:
            return ""
        _, _, caption_block = figure.partition(r"\caption{")
        caption, unmatched = "", 1
        for c in caption_block:
            if c == "{":
                unmatched += 1
            elif c == "}":
                unmatched -= 1
            if unmatched == 0:
                break
            caption += c
        return caption.strip()
    except Exception:
        return ""


def remove_unmatched_delimiters(text, open_delim, close_delim):
    open_count = text.count(open_delim)
    close_count = text.count(close_delim)
    if open_count > close_count:
        for _ in range(open_count - close_count):
            idx = text.rfind(open_delim)
            if idx != -1:
                text = text[:idx] + text[idx+len(open_delim):]
    elif close_count > open_count:
        for _ in range(close_count - open_count):
            idx = text.rfind(close_delim)
            if idx != -1:
                text = text[:idx] + text[idx+len(close_delim):]
    return text


def remove_unmatched_unescaped(text, open_char, close_char):
    opens = len(findall(rf'(?<!\\){escape(open_char)}', text))
    closes = len(findall(rf'(?<!\\){escape(close_char)}', text))
    if opens > closes:
        for _ in range(opens - closes):
            idx = text.rfind(open_char)
            if idx != -1 and text[idx-1] != '\\':
                text = text[:idx] + text[idx+1:]
    elif closes > opens:
        for _ in range(closes - opens):
            idx = text.rfind(close_char)
            if idx != -1 and text[idx-1] != '\\':
                text = text[:idx] + text[idx+1:]
    return text


def count_unescaped_dollars(text):
    return len(findall(r'(?<!\\)\$', text))


def remove_unmatched_dollars(text):
    unescaped_dollars = count_unescaped_dollars(text)
    if unescaped_dollars % 2 != 0:
        parts = list(finditer(r'(?<!\\)\$', text))
        if parts:
            idx = parts[-1].start()
            text = text[:idx] + text[idx+1:]
    return text


def pre_clean_math_delimiters(text):
    text = remove_unmatched_dollars(text)
    text = remove_unmatched_delimiters(text, r"\(", r"\)")
    text = remove_unmatched_unescaped(text, '{', '}')
    return text


def clean_caption(caption, macros):
    expanded = expand_macros(macros, caption)
    cleaned = pre_clean_math_delimiters(expanded)
    try:
        cap_soup = TexSoup(cleaned, tolerance=1)
        for label in cap_soup.find_all("label"):
            label.delete()
        return " ".join(str(cap_soup).split())
    except Exception:
        return " ".join(cleaned.split())


def lines_startwith(text, prefix):
    return all(line.lstrip().startswith(prefix) for line in text.strip().splitlines() if line.strip())


def lines_removeprefix(text, prefix):
    return "\n".join(line.lstrip()[len(prefix):] if line.lstrip().startswith(prefix) else line for line in text.splitlines())


def clean_code(code, prefix):
    try:
        code = clean_tikz_code(code)
        code = dedent(code.strip())
        prefix = clean_tikz_code(prefix)
        prefix = dedent(prefix.strip())
        if lines_startwith(code, "%"):
            return dedent(lines_removeprefix(code, "%"))
        elif prefix and prefix in code:
            return code.replace(prefix, "", 1).strip()
        return code
    except Exception:
        return code


def replace_graphics(tikz_code):
    try:
        graphicx_regex = r'(\\includegraphics\*?(?:\[.*?\]){0,2})\{.+?\}'
        return sub(graphicx_regex, r'\1{example-image}', tikz_code)
    except Exception:
        return tikz_code


def remove_external_dependencies(code):
    for pattern in EXTERNAL_DEPENDENCY_PATTERNS:
        code = sub(pattern, '', code, flags=MULTILINE)
    return code


def remove_empty_lines(code):
    return "\n".join([line for line in code.splitlines() if line.strip()])


def trim_environment_block(code, env):
    try:
        code = sub(rf"(?s)^.*?\\begin{{{env}}}", rf"\\begin{{{env}}}", code)
        code = sub(rf"\\end{{{env}}}.*$", rf"\\end{{{env}}}", code)
        return code.strip()
    except Exception:
        return code


def build_tikz_document(env, tikz_code, imports, macros):
    try:
        if not isinstance(tikz_code, str):
            return ""
        tikz_code = tikz_code.strip()
        if SAME_ENV:
            preamble = r"\documentclass[tikz]{standalone}"
            clean_imports = []
            for line in imports.splitlines():
                line = line.strip()
                if line.startswith(r"\documentclass"):
                    continue
                if any(line.startswith(prefix) for prefix in (r"\usepackage", r"\usetikzlibrary", r"\tikzset", r"\tikzstyle", r"\newcommand", r"\def")):
                    clean_imports.append(line)
            if not clean_imports and not OTHER_IMG:
                clean_imports = DEFAULT_PREAMBLE
            if REM_EXT:
                tikz_code = remove_external_dependencies(tikz_code)
                tikz_code = remove_empty_lines(tikz_code)
            colors = extract_color_definitions(macros, tikz_code)
            if colors:
                preamble += f"\n\n{colors}"
            used_macros = expand_macros(macros, tikz_code, expand=False)
            if used_macros:
                preamble += f"\n\n{used_macros}"
            if env:
                tikz_code = trim_environment_block(tikz_code, env)
            return "\n\n".join([
                preamble,
                "\n".join(clean_imports).strip(),
                r"\begin{document}",
                tikz_code.strip(),
                r"\end{document}"
            ])
        else:
            tikz_code = replace_graphics(tikz_code)
            used_macros = expand_macros(macros, tikz_code, expand=False)
            colors = extract_color_definitions(macros, tikz_code)
            preamble = imports
            if colors:
                preamble += f"\n\n{colors}"
            if used_macros:
                preamble += f"\n\n{used_macros}"
            return "\n\n".join([preamble, r"\begin{document}", tikz_code, r"\end{document}"])
    except Exception:
        return ""
    

def extract_prefix(text, idx):
    head = text[:idx].split("\\begin{document}")[0]
    lines = head.splitlines()
    seen = set()
    pkg_lines = []
    for l in lines:
        if search(r"\\(?:usepackage|usetikzlibrary|newcommand|def|tikzset|tikzstyle)", l):
            l = l.strip()
            if l and l not in seen:
                pkg_lines.append(l)
                seen.add(l)
    if pkg_lines:
        return "\n".join(["\\documentclass[tikz]{standalone}"] + pkg_lines)
    return DEFAULT_PREAMBLE


def find_surrounding_figure_caption(tex, code_start, code_end):
    figure_env_regex = re.compile(r"\\begin\{figure\}(.*?)\\end\{figure\}", DOTALL)
    for match in figure_env_regex.finditer(tex):
        f_start, f_end = match.span()
        if f_start <= code_start and code_end <= f_end:
            block = match.group(1)
            caption_match = search(r"\\caption\{(.*?)\}", block, DOTALL)
            label_match = re.search(r"\\label\{(.*?)\}", block, DOTALL)
            caption = caption_match.group(1).strip() if caption_match else ""
            label = label_match.group(1).strip() if label_match else ""
            return caption, label
    return "", ""


def extract_figures_with_tikz(tex):
    try:
        if OTHER_IMG:
            env_pattern = rf"\\begin\{{({'|'.join(SUPPORTED_ENVIRONMENTS)})\}}(\[[^\]]*\])?"
            pattern = re.compile(env_pattern + r".*?\\end\{\1\}", DOTALL)
            for match in pattern.finditer(tex):
                code = match.group(0).strip()
                if len(code) > MAX_TIKZ_LENGTH:
                    continue
                env = match.group(1)
                start, end = match.span()
                prefix = extract_prefix(tex, start)
                caption, label = find_surrounding_figure_caption(tex, start, end)
                yield env, code, prefix, caption, label
        else:
            env = 'tikzpicture'
            figure_regex = r"\\begin{figure}(.*?)\\end{figure}"
            tikz_regex = r"(([^\n]*?)\\begin{tikzpicture}.*?\\end{tikzpicture})"
            for figure in findall(figure_regex, tex, DOTALL):
                if figure.count(r"\begin{tikzpicture}") == 1:
                    tikz = search(tikz_regex, figure, DOTALL)
                    if tikz:
                        code = tikz.group()
                        if len(code) > MAX_TIKZ_LENGTH:
                            continue
                        caption = extract_caption(figure)
                        label = extract_label(figure)
                        yield env, tikz.group(), tikz.group(2), caption, label
            for tikz, prefix in findall(tikz_regex, tex, DOTALL):
                yield env, tikz, prefix, "", ""
    except Exception:
        pass


class SingleTikzProcessor:
    def __init__(self, code, caption):
        self.tex = code
        self.caption = caption
        self.finder = TikzFinder(tex=code)
        self.imports, self.macros = extract_preamble(self.tex)

    def process(self):
        results = []
        for env, tikz_code, prefix, cap_raw, _ in extract_figures_with_tikz(self.tex):
            cleaned_code = clean_code(tikz_code, prefix)
            caption = clean_caption(self.caption or cap_raw, self.macros)
            full_doc = build_tikz_document(env, cleaned_code, self.imports, self.macros)
            results.append({
                "figure_id": None,
                "caption": self.caption,
                "code": full_doc
            })
        return results
    
def process_individual_json_entries(json_entries):
    tikz_outputs = []
    for entry in json_entries:
        code = entry.get("code", "")
        caption = entry.get("caption", "")
        fig_id = entry.get("figure_id", None)
        processor = SingleTikzProcessor(code, caption)
        result = processor.process()
        for r in result:
            r["figure_id"] = fig_id
        tikz_outputs.extend(result)
    return tikz_outputs


class TikzFinder:
    def __init__(self, tex):
        self.tex = self._validate(tex)
        self.imports, self.macros = extract_preamble(self.tex)

    def _validate(self, tex):
        try:
            tex = tex.strip()
            has_docclass = r"\documentclass" in tex
            has_begin = r"\begin{document}" in tex
            has_end = r"\end{document}" in tex
            if not has_docclass and not has_begin and not has_end:
                return "\n".join([
                    DEFAULT_PREAMBLE.strip(),
                    r"\begin{document}",
                    tex,
                    r"\end{document}"
                ])
            if not has_docclass:
                tex = DEFAULT_PREAMBLE.strip() + "\n\n" + tex
            if not has_begin or not has_end:
                parts = tex.split(r"\end{document}", 1)
                tex = r"\begin{document}" + "\n" + parts[0] + "\n" + r"\end{document}"
                if len(parts) > 1:
                    tex += "\n" + parts[1]
            return tex.strip()
        except Exception:
            return "\n".join([
                DEFAULT_PREAMBLE.strip(),
                r"\begin{document}",
                tex if isinstance(tex, str) else "",
                r"\end{document}"
            ])

    def find(self):
        found = set()
        for env, tikz_code, prefix, caption_raw, label in extract_figures_with_tikz(self.tex):
            cleaned_code = clean_code(tikz_code, prefix)
            if cleaned_code in found:
                continue
            found.add(cleaned_code)
            caption = clean_caption(caption_raw, self.macros)
            full_doc = build_tikz_document(env, cleaned_code, self.imports, self.macros)
            yield Tikz(full_doc, caption, label)

    def __call__(self):
        yield from self.find()


def _load_worker(args):
    paper, base_name, chunk_idx, paper_idx = args
    found = []
    try:
        text = paper.get("text", "")
        paper_meta = paper.get("meta", "")
        if TEXT_MENTION:
            results = _process_tikz_with_text_mention(text)
        else:
            results = _process_tikz(text)
        for tikz_idx, tikz in enumerate(results):
            result = {
                "caption": tikz["caption"],
                "code": tikz["code"],
                "file_id": f"{base_name}_{chunk_idx}_{paper_idx}_{tikz_idx}",
                "arxiv_id": paper_meta.get("arxiv_id"),
                "url": paper_meta.get("url")
            }
            if TEXT_MENTION and "text_mentions" in tikz:
                result["text_mentions"] = tikz["text_mentions"]
            found.append(result)
    except TimeoutError:
        pass
    except (AssertionError, RecursionError):
        pass
    except Exception:
        pass
    return found


def _process_tikz(text):
    return [
        {"caption": tikz.caption, "code": tikz.code, "label": tikz.label}
        for tikz in TikzFinder(tex=text).find()
    ]


def _process_tikz_with_text_mention(text):
    results = []
    tikz_finder = TikzFinder(tex=text)
    figures = list(tikz_finder.find())
    label_to_tikz = {
        tikz.label: tikz
        for tikz in figures
        if tikz.label
    }
    mentions = extract_text_mentions(text, tikz_labels=list(label_to_tikz.keys()))
    for tikz in figures:
        entry = {
            "caption": tikz.caption,
            "code": tikz.code
        }
        if TEXT_MENTION and tikz.label and tikz.label in mentions:
            entry["text_mentions"] = mentions[tikz.label]
        results.append(entry)
    return results


def expand(files):
    for file in files:
        if os.path.isdir(file):
            yield from glob(join(file, "*.jsonl"))
        else:
            yield file


class TimeoutException(Exception): pass


def handler(signum, frame): raise TimeoutException()


def _load_worker_with_timeout(args, timeout=TIMEOUT):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        return _load_worker(args)
    except TimeoutException:
        return []
    finally:
        signal.alarm(0)

def safe_worker(args):
    try:
        return _load_worker_with_timeout(args)
    except Exception:
        return []


def extract_tikz(files, num_workers, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_files = list(expand(files))
    for idx, file in enumerate(all_files):
        base_name = Path(file).stem
        output_file = os.path.join(output_dir, f"{base_name}.json")
        if os.path.exists(output_file):
            continue
        try:
            chunk = load_dataset("json", data_files=file, split="train")
            if len(chunk) == 0:
                continue
            inputs = [(paper, base_name, idx, i) for i, paper in enumerate(chunk)]
            tikz_entries = []
            with get_context("spawn").Pool(processes=num_workers, maxtasksperchild=10) as pool:
                result_iter = pool.imap_unordered(safe_worker, inputs)
                for result in tqdm(result_iter, total=len(inputs), start=1):
                    tikz_entries.extend(result)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tikz_entries, f, indent=4, ensure_ascii=False)
        except Exception:
            continue


if __name__ == "__main__":
    base_name = "arxiv"  # "arxiv" or "github"
    input_path = [f"datikz/{base_name}_extracted"]
    output_dir = f"datikz/{base_name}_output"
    num_workers = len(os.sched_getaffinity(0))
    existing = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    extract_tikz(input_path, num_workers, output_dir)