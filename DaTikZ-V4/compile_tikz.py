import os
import re
import json
import shutil
import tarfile
import tempfile
import subprocess

from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps
from pdfCropMargins import crop
from multiprocessing import Pool


SIZE = 448
LATEX_TIMEOUT = 120
COMPILERS = ['pdflatex', 'lualatex', 'xelatex']


def extract_error_section(log_text, max_lines=30, context_lines=3):
    if not log_text:
        return ""
    match = re.search(r"^! .*", log_text, re.MULTILINE)
    if match:
        i = log_text[:match.start()].count('\n')
        lines = log_text.splitlines()
        start = max(i - context_lines, 0)
        end = min(i + max_lines, len(lines))
        return '\n'.join(lines[start:end])
    return '\n'.join(log_text.splitlines()[-max_lines:])


def process_figure(index, fig, png_dir):
    file_id = fig['file_id']
    latex_code = fig['code_new']
    result_entry = {"index": index, "status": None, "log": None}
    with tempfile.TemporaryDirectory() as tmpdirname:
        work_dir = Path(tmpdirname)
        tex_file = work_dir / 'figure.tex'
        pdf_file = work_dir / 'figure.pdf'
        latex_lines = latex_code.splitlines()
        latex_lines.insert(1, r"\AtBeginDocument{\thispagestyle{empty}\pagestyle{empty}}")
        tex_file.write_text('\n'.join(latex_lines), encoding='utf-8')
        try:
            tex_file.with_suffix(".bbl").touch(exist_ok=True)
        except Exception:
            pass
        compiled = False
        first_log = None
        # open(f"{tex_file}.bbl", 'a').close()
        for j, compiler in enumerate(COMPILERS):
            try:
                result = subprocess.run(
                    [compiler, '-interaction=nonstopmode', '-halt-on-error', str(tex_file)],
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=LATEX_TIMEOUT)
                log_output = result.stdout.decode('utf-8', errors='replace')
                if j == 0:
                    first_log = log_output
                if pdf_file.exists() and pdf_file.stat().st_size > 0:
                    compiled = True
                    break
            except subprocess.SubprocessError as e:
                if j == 0:
                    first_log = str(e)
            except subprocess.TimeoutExpired as e:
                if j == 0:
                    first_log = (e.stdout or b"").decode("utf-8", errors="replace")
        if not compiled:
            result_entry["status"] = "error"
            result_entry["log"] = extract_error_section(first_log)
            return result_entry
        try:
            cropped_pdf = pdf_file.with_name(pdf_file.stem + '_cropped.pdf')
            crop(["-c", "gb", "-p", "0", "-a", "-1", "-o", str(cropped_pdf), str(pdf_file)], quiet=True)
            if cropped_pdf.exists() and cropped_pdf.stat().st_size > 0:
                pdf_file = cropped_pdf
        except Exception:
            pass
        try:
            subprocess.run(
                ['pdftoppm', '-singlefile', '-png', str(pdf_file), str(pdf_file.with_suffix(''))],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60)
            png_path_tmp = pdf_file.with_suffix('.png')
            image = Image.open(png_path_tmp)
            if image.getcolors(1) is not None:
                result_entry["status"] = "empty"
                result_entry["log"] = extract_error_section(first_log)
                return result_entry
            try:
                image = ImageOps.pad(image, (SIZE, SIZE), color='white')
            except Exception:
                pass
            out_png = png_dir / f"{file_id}.png"
            image.save(out_png)
            result_entry["status"] = "success"
            return result_entry
        except subprocess.TimeoutExpired:
            result_entry["status"] = "error"
            result_entry["log"] = "pdftoppm timeout"
            return result_entry
        except Exception as e:
            result_entry["status"] = "error"
            result_entry["log"] = extract_error_section(first_log)
            return result_entry


def process_figure_wrapper(args):
    index, fig, png_dir = args
    return process_figure(index, fig, png_dir)


def archive_folder(png_dir):
    png_archive = png_dir.parent / f"{png_dir.name}.tar.gz"
    with tarfile.open(png_archive, "w:gz") as tar:
        tar.add(png_dir, arcname=".")
    return png_archive


def arxiv_index(path):
    return int(path.stem.split("_")[-1])


if __name__ == "__main__":
    num_workers = len(os.sched_getaffinity(0))
    for json_path in sorted(Path("outputs").glob("arxiv_*.json"), key=arxiv_index):
        stem = json_path.stem
        png_dir = Path("outputs") / "pngs_new" / stem
        png_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        args_list = [
            (i, fig, png_dir)
            for i, fig in enumerate(all_data)
            if fig.get("status") != "success"
            and fig.get("code_new", "") != fig.get("code", "")
        ]
        if args_list:
            with Pool(processes=num_workers, maxtasksperchild=1) as pool:
                it = pool.imap_unordered(
                    process_figure_wrapper,
                    args_list,
                    chunksize=1,
                )
                results = list(tqdm(it, total=len(args_list)))
            for result in results:
                all_data[result["index"]].update({
                    "status_new": result["status"],
                    "log_new": result.get("log", ""),
                })
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
        archive_folder(png_dir)
        shutil.rmtree(png_dir)