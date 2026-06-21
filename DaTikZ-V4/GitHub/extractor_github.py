import os
import json
import tarfile
import tempfile
import subprocess

from pathlib import Path
from subprocess import DEVNULL
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed


TARBALL_DIR = "GitHub/repos"
OUTPUT_DIR = "GitHub/repos_processed"
TIKZ_MARKER = "tikz"
MAX_WORKERS = 4
MAX_FAILURES = 10
LATEXPAND_TIMEOUT = 90


def find_tex_files(root_dir):
    return [str(p) for ext in ("*.tex", "*.pgf") for p in Path(root_dir).rglob(ext)]


def latexpand(tex_file_path):
    with NamedTemporaryFile(buffering=0) as tmp:
        path = os.path.dirname(tex_file_path) or "."
        file = os.path.basename(tex_file_path)
        cmd = ["latexpand", "--keep-comments", file, "--output", tmp.name]
        try:
            subprocess.run(cmd, cwd=path, stdout=DEVNULL, stderr=DEVNULL, check=True, timeout=LATEXPAND_TIMEOUT)
            tmp.seek(0)
            return tmp.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return False


def process_repo(repo_fullname):
    try:
        failures = 0
        repo_filename = repo_fullname.replace("/", "__")
        output_path = os.path.join(OUTPUT_DIR, f"{repo_filename}.jsonl")
        tarball_path = os.path.join(TARBALL_DIR, f"{repo_filename}.tar.gz")
        if os.path.exists(output_path):
            return
        if not os.path.exists(tarball_path):
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with tarfile.open(tarball_path, "r:gz") as tar:
                    tar.extractall(path=tmpdir)
            except Exception:
                return
            tex_files = find_tex_files(tmpdir)
            tikz_snippets = []
            for tex in tex_files:
                if failures > MAX_FAILURES:
                    return
                expanded = latexpand(tex)
                content_to_check = expanded
                if expanded is False:
                    failures += 1
                    try:
                        with open(tex, "r", encoding="utf-8", errors="replace") as f:
                            content_to_check = f.read()
                    except Exception:
                        continue
                if content_to_check and TIKZ_MARKER in content_to_check:
                    tikz_snippets.append({
                        "path": os.path.relpath(tex, tmpdir),
                        "content": content_to_check
                    })
            if tikz_snippets:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for snippet in tikz_snippets:
                        f.write(json.dumps(snippet, ensure_ascii=False) + "\n")
                return
            else:
                return
    except Exception:
        return


def runner(args):
    repo_name, _ = args
    return process_repo(repo_name)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tarballs = sorted(Path(TARBALL_DIR).glob("*.tar.gz"))
    remaining_tarballs = []
    for tarball in tarballs:
        repo_name = tarball.stem.replace(".tar", "")
        output_path = os.path.join(OUTPUT_DIR, f"{repo_name}.jsonl")
        if not os.path.exists(output_path):
            remaining_tarballs.append((repo_name, tarball))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="repo-worker") as executor:
        futures = {executor.submit(runner, pair): pair[0] for pair in remaining_tarballs}
        for future in as_completed(futures):
            print(future.result())