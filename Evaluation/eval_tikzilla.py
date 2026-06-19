import os
import json
import base64
import hashlib
import tempfile
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pdfCropMargins import crop
from PIL import Image, ImageOps
from tiktoken.core import Encoding

from APIs.llms_api import Gpt4Api, Gpt5Api, QwenApi, Llama3_1Api, Qwen3CoderApi
from Metrics.evaluation_metrics import BatchedClipScore, BatchedClipScoreImg, BatchedDeTikZifyScore, CrystalBLEU, DreamSim, TexEditDistance

os.environ["PATH"] = f"{os.path.expanduser('/home/hpc/<USERNAME>/texlive/bin/x86_64-linux')}:" + os.environ["PATH"]
pdf_to_ppm_path = "/home/hpc/<USERNAME>/poppler-24.07.0/build/utils/pdftoppm"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--work_dir", required=True)
    return parser.parse_args()

def load_text_tiktoken_vocab(filepath):
    mergeable_ranks = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            b64_token, token_id_str = line.strip().split()
            token_bytes = base64.b64decode(b64_token)
            token_id = int(token_id_str)
            mergeable_ranks[token_bytes] = token_id
    return mergeable_ranks

def process_figure(png_dir, figure_id, tikz_code):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tempfile.NamedTemporaryFile(dir=tmpdirname, buffering=0) as tmpfile:
            work_dir = Path(tmpdirname)
            tex_file = work_dir / 'figure.tex'
            pdf_file = work_dir / 'figure.pdf'
            
            latex_lines = tikz_code.splitlines()
            latex_lines.insert(1, r"\AtBeginDocument{\thispagestyle{empty}\pagestyle{empty}}")
            tex_file.write_text('\n'.join(latex_lines), encoding='utf-8')

            compiled = False
            open(f"{tex_file}.bbl", 'a').close()
            for compiler in ['pdflatex', 'lualatex', 'xelatex']:
                try:
                    subprocess.run(
                        [compiler, '-interaction=nonstopmode', '-halt-on-error', str(tex_file)],
                        cwd=work_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=120
                    )
                    if pdf_file.exists() and pdf_file.stat().st_size > 0:
                        compiled = True
                        break
                except subprocess.SubprocessError as e:
                    print(f"Subprocess error: {e}")

            if not compiled:
                return False
            
            try:
                cropped_pdf = pdf_file.with_name(pdf_file.stem + '_cropped.pdf')
                crop(["-c", "gb", "-p", "0", "-a", "-1", "-o", str(cropped_pdf), str(pdf_file)], quiet=True)
                if cropped_pdf.exists():
                    pdf_file = cropped_pdf
            except Exception as e:
                print(f"Cropping error: {e}")
            try:
                subprocess.run(
                    [pdf_to_ppm_path, '-singlefile', '-png', str(pdf_file), str(pdf_file.with_suffix(''))],
                    check=True
                )
                png_path = pdf_file.with_suffix('.png')
                image = Image.open(png_path)

                if image.getcolors(1) is not None:
                    return False

                try:
                    image = ImageOps.pad(image, (448, 448), color='white')
                except Exception as e:
                    print(f"Image resizing error: {e}")

                png_path = png_dir / f"{figure_id}.png"
                image.save(png_path)
                return png_path

            except Exception as e:
                print(f"Rasterization error: {e}")
    return False

def construct_prompt_tikz_code(query):
    structure = """
    Generate a complete LaTeX document that contains a TikZ figure according to the following requirements: 
    {query} 
    Wrap your code using \\documentclass[tikz]{{standalone}}, and include \\begin{{document}}...\\end{{document}}. 
    Only output valid LaTeX code with no extra text.
    """
    return structure.format(query=query)

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

def extract_tikz(raw_output):
    first_backslash = raw_output.find('\\')
    if first_backslash == -1:
        return ''
    end_tag = '\\end{document}'
    end_index = raw_output.find(end_tag)
    if end_index == -1:
        end_index = raw_output.rfind('\\')
        return raw_output[first_backslash:end_index+1].strip()
    return raw_output[first_backslash:end_index + len(end_tag)].strip()

def save_batch(output_path, batch):
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(batch, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args = arg_parser()
    bpe_path = "models/o200k_base.tiktoken"
    with open(bpe_path, "rb") as f:
        contents = f.read()
    expected_hash = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    file_hash = hashlib.sha256(contents).hexdigest()
    if file_hash != expected_hash:
        raise ValueError(f"Hash mismatch: got {file_hash}")
    mergeable_ranks = load_text_tiktoken_vocab(bpe_path)
    pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^ \p{L}\p{N}]+|\s+(?!\S)|\s+"""
    encoding = Encoding(
        name="gpt-4o-local",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={}
    )
    clip_scorer_batched = BatchedClipScore(model_name="models/siglip-so400m-patch14-384")
    clip_scorer_img_batched = BatchedClipScoreImg(model_name="models/siglip-so400m-patch14-384")
    detikzify_scorer_batched = BatchedDeTikZifyScore(model_name=f"{args.work_dir}/models/detikzify-v2-8b")
    crystal_bleu_scorer = CrystalBLEU(corpus=None, cache_key="crystalbleu_tikz_5k", only_code=True)
    dreamsim_scorer = DreamSim(model_name="ensemble")
    tex_edit_scorer = TexEditDistance(only_code=True)

    finetuned = False
    if args.model_id == "Qwen2.5-3B-finetuned-full" or args.model_id == "Qwen2.5-3B-finetuned-lora" or args.model_id == "Qwen3-8B-Base-finetuned-full" or args.model_id == "Qwen3-8B-Base-finetuned-lora":
        finetuned = True
    adapter_name = "Qwen3-8B-Base"
    checkpoint_name = "checkpoint-10000"
    adapter_path = f"{args.work_dir}/trained_models_grpo/{adapter_name}/{checkpoint_name}" #trained_models_sft, trained_models_grpo

    input_path_static = "captions_new/test_data_gpt4o.json"
    input_path = f"captions_new/{args.input_file}"
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    pred_img_dir = Path("captions_new/test_data")

    if finetuned:
        output_dir = Path(f"evaluations/{input_basename}_{adapter_name}_{checkpoint_name}")
        output_file = f"evaluations/{input_basename}_{adapter_name}_{checkpoint_name}.json"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = Path(f"evaluations/{input_basename}_{args.model_id}")
        output_file = f"evaluations/{input_basename}_{args.model_id}.json"
        os.makedirs(output_dir, exist_ok=True)

    if args.model_id == "Llama-3.1-8B-Instruct":
        model_tikz = Llama3_1Api(work_dir=args.work_dir, model_id=args.model_id, temperature=1.0, top_p=0.9)
    if args.model_id == "Qwen3-Coder-30B-A3B-Instruct":
        model_tikz = Qwen3CoderApi(work_dir=args.work_dir, model_id=args.model_id, temperature=1.0, top_p=0.9)
    if args.model_id == "gpt-4o" or args.model_id == "gpt-4o-mini":
        model_tikz = Gpt4Api(model_id=args.model_id, temperature=1.0, top_p=0.9)
    if args.model_id == "gpt-5":
        model_tikz = Gpt5Api(model_id=args.model_id, temperature=1.0, top_p=0.9)
    if args.model_id == "Qwen2.5-3B" or args.model_id == "Qwen2.5-3B-finetuned-full" or args.model_id == "Qwen2.5-3B-finetuned-lora" or args.model_id == "Qwen3-8B-Base" or args.model_id == "Qwen3-8B-Base-finetuned-full" or args.model_id == "Qwen3-8B-Base-finetuned-lora":
        model_tikz = QwenApi(work_dir=args.work_dir, model_id=args.model_id, adapter_path=adapter_path, finetuned=finetuned, temperature=1.0, top_p=0.9)

    with open(input_path_static, "r", encoding="utf-8") as f:
        all_data_gt = json.load(f)
    if input_path.endswith(".jsonl"):
        all_new_data_gt = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_new_data_gt.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
    
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                processed_queries = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing {output_file}. Starting fresh.")
                processed_queries = []
    else:
        processed_queries = []

    tikz_pred_all = []
    tikz_gt_all = []
    png_path_pred_all = []
    png_path_gt_all = []

    tex_edit_scores = []
    dreamsim_scores = []

    idx = 0
    not_compiled_idx = 0

    compiled = None

    for entry_gt in tqdm(all_data_gt):
        idx += 1
        figure_id = entry_gt.get("figure_id")
        tikz_gt = entry_gt.get("code")
        if input_path.endswith(".json"):
            description_gt = entry_gt.get("new_caption")
        elif input_path.endswith(".jsonl"):
            description_gt = next((entry["new_caption"] for entry in all_new_data_gt if entry.get("file_id") == figure_id), None)

        try:
            response = model_tikz.request(description_gt)
        except Exception as e:
            continue
        tikz_code = extract_tikz(response)
        tikz_pred = clean_tikz_code(tikz_code)

        png_path_pred = process_figure(output_dir, figure_id, tikz_pred)

        if png_path_pred:
            png_path_pred_all.append(png_path_pred)
            png_path_gt = pred_img_dir / f"{figure_id}.png"
            png_path_gt_all.append(png_path_gt)
            dreamsim_score = dreamsim_scorer(png_path_pred, png_path_gt)
            compiled = True
        elif not png_path_pred:
            dreamsim_score = 0.0
            not_compiled_idx += 1
            compiled = False

        dreamsim_scores.append(dreamsim_score)
        tikz_pred_all.append(tikz_pred)
        tikz_gt_all.append(tikz_gt)
        tex_edit_score = tex_edit_scorer(tikz_pred, tikz_gt)
        tex_edit_scores.append(tex_edit_score)

        processed_queries.append({
            "figure_id": figure_id,
            "description": description_gt,
            "tikz_pred": tikz_pred,
            "dream_sim": dreamsim_score,
            "tex_edit_distance": tex_edit_score,
            "compiled": compiled
        })

        save_batch(output_file, processed_queries)

    crystal_bleu_scores = crystal_bleu_scorer(tikz_pred_all, tikz_gt_all)

    clip_scores = clip_scorer_batched(png_path_pred_all, png_path_gt_all)
    clip_scores += [0.0] * (168 - len(clip_scores))
    clip_scores_img = clip_scorer_img_batched(png_path_pred_all, png_path_gt_all)
    clip_scores_img += [0.0] * (168 - len(clip_scores_img))
    detikzify_scores = detikzify_scorer_batched(png_path_pred_all, png_path_gt_all)
    detikzify_scores += [0.0] * (168 - len(detikzify_scores))

    print(f"CLIPScore (Text2Image): {np.average(clip_scores)}")
    print(f"CLIPScore (Image2Image): {np.average(clip_scores_img)}")
    print(f"DeTikZifyScore: {np.average(detikzify_scores)}")
    print(f"CrystalBLEU: {crystal_bleu_scores}")
    print(f"DreamSim: {np.average(dreamsim_scores)}")
    print(f"TexEditDistance: {np.average(tex_edit_scores)}")
    print(f"Average Score (CLIP, DSim, TED): {(np.average(clip_scores) + np.average(dreamsim_scores) + (1 - np.average(tex_edit_scores))) / 3}")

    print(f"Compilation Rate: {(idx - not_compiled_idx) / idx}")
    token_efficiency_scores = [len(encoding.encode(tikz_pred)) for tikz_pred in tikz_pred_all]
    print(f"Token Efficiency: {np.average(token_efficiency_scores)} +- {np.std(token_efficiency_scores)}")