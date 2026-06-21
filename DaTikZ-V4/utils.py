import re

from contextlib import contextmanager
from datasets import (disable_progress_bar, enable_progress_bar, is_progress_bar_enabled)


def remove_latex_comments(line):
    result = []
    i = 0
    while i < len(line):
        if line[i] == '%':
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                break
            else:
                result.append('%')
        else:
            result.append(line[i])
        i += 1
    return ''.join(result).rstrip()


def clean_tikz_code(tikz_raw):
    cleaned_lines = []
    for line in tikz_raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%'):
            continue
        cleaned_line = remove_latex_comments(line)
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines).strip()


def is_non_visual_or_uncompilable_tikz(code):
    code_lower = code.lower()
    if re.search(r'\\usetikzlibrary\{[^}]*animations[^}]*\}', code_lower):
        return True
    if re.search(r'\\usepgfmodule\{[^}]*animations[^}]*\}', code_lower):
        return True
    if re.search(r'\\pgfmathprintnumber\s*\{[^}]+\}', code):
        return True
    return False


def lines_startwith(string, prefix):
    return all(line.startswith(prefix) for line in string.splitlines())


def lines_removeprefix(string, prefix):
    return "".join(line.removeprefix(prefix) for line in string.splitlines(keepends=True))


@contextmanager
def no_progress_bar():
    if is_progress_bar_enabled():
        try:
            yield disable_progress_bar()
        finally:
            enable_progress_bar()