import os
import re
import sys
import uuid
import torch
import shutil
import hashlib
import argparse
import subprocess
import numpy as np
import multiprocessing as mp

from ot.lp import emd2
from shutil import which
from pathlib import Path
from dreamsim import dreamsim
from PIL import Image, ImageOps
from pdfCropMargins import crop
from detikzify.util import expand
from collections import OrderedDict
from trl import GRPOConfig, GRPOTrainer
from detikzify.model import load as load_model
from Data_Processing.tokenization import make_qwen_base_preprocessor_grpo
from datasets import concatenate_datasets, load_from_disk
from torchmetrics.functional import pairwise_cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint


os.environ["PATH"] = f"{os.path.expanduser('/home/hpc/<USERNAME>/texlive/bin/x86_64-linux')}:" + os.environ["PATH"]
PDFTOPPM = "/home/hpc/<USERNAME>/poppler-24.07.0/build/utils/pdftoppm"
GS_BIN = "/home/hpc/<USERNAME>/ghostscript-10.03.0/bin"
os.environ["PATH"] = f"{GS_BIN}:" + os.environ["PATH"]


def arg_parser():
    parser = argparse.ArgumentParser(description="Finetune LLMs on TikZ code (GRPO).")
    parser.add_argument('--image_reward_backend', type=str, default="detikzify", choices=["detikzify", "clip", "dreamsim"], help="Which image encoder to use for reward computation.")
    parser.add_argument('--model_type', type=str, default="Qwen2.5-3B", help="Model type of the LLM to be finetuned.") #Qwen2.5-3B, Qwen3-8B-Base
    parser.add_argument('--model_id', type=str, default="TikZilla-3B", help="ID of the LLM to be finetuned.") #TikZilla-3B, TikZilla-8B
    parser.add_argument('--base_model', type=bool, default=False, help="Uses a base or already finetuned model")
    parser.add_argument('--checkpoint_id', type=str, default="checkpoint-27000", help="ID of the checkpoint to be used.")
    parser.add_argument('--reward_model_id', type=str, default="detikzify-v3-8b", help="ID of the model for calculating rewards.") #detikzify-v2-8b, detikzify-v3-8b, siglip-so400m-patch14-384, ensemble
    parser.add_argument('--checkpoint_id_reward_model', type=str, default="checkpoint-20000", help="Reward Model ID of the checkpoint to be used.")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length of input + output.")
    parser.add_argument('--code_length', type=tuple, default=(100, 4000), help="Min and max TikZ code lengths.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for finetuning.")
    parser.add_argument('--device_batch_size', type=int, default=6, help="Batch size per device.") #6, 3
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6, help="Number steps for gradient accumulation.") #6, 12
    parser.add_argument('--num_generations', type=int, default=8, help="Number of generations for GRPO.")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate for finetuning.") #5e-6, 2e-6
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Value for gradient clipping.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="Warmup percentage of sheduler.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature value for sampling.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top p value for sampling.")
    parser.add_argument('--epsilon', type=float, default=0.2, help="GRPO clipping hyperparameter.")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay to prevent overfitting.")
    parser.add_argument('--epsilon_high', type=float, default=0.28, help="Different epsilon parameter.")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help="Learning rate sheduler type.")
    parser.add_argument('--loss_type', type=str, default="dr_grpo", help="Type of loss function for GRPO.")
    parser.add_argument('--mask_truncated_completions', type=bool, default=True, help="Masks completions that are too long.")
    parser.add_argument('--work_dir', type=str, required=True, help="Path to work dir.")
    parser.add_argument('--tmp_dir', type=str, required=True, help="Path to tmp dir.")
    return parser.parse_args()


_DOC_RE = re.compile(
    r"""
    (?P<prefix>\\documentclass\s*\[[^\]]*?\btikz\b[^\]]*?\]\s*\{standalone\})
    (?P<mid>.*?)
    (\\begin\{document\})
    (?P<body>.*?)
    (\\end\{document\})
    """,
    re.DOTALL | re.VERBOSE
)


def _extract_tikz_doc(text):
    if not isinstance(text, str): 
        return None
    cleaned = text.replace("\x00", "")
    m = _DOC_RE.search(cleaned)
    if not m: return None
    return cleaned[m.start("prefix"):m.end()].strip()


def crop_pdf_out_of_proc(pdf_in, pdf_out):
    use_gs = which("gs") is not None
    cmd = [sys.executable, "-m", "pdfCropMargins"]
    if use_gs:
        cmd += ["-c", "gb"]
    cmd += ["-p", "0", "-a", "-1", "-o", str(pdf_out), str(pdf_in)]
    subprocess.run(cmd, cwd=Path(pdf_in).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def process_figure_in_dir(work_dir, figure_id, tikz_code):
    work_dir.mkdir(parents=True, exist_ok=True)
    tex_file = work_dir / f"{figure_id}.tex"
    pdf_file = work_dir / f"{figure_id}.pdf"
    lines = tikz_code.splitlines()
    lines.insert(1, r"\AtBeginDocument{\thispagestyle{empty}\pagestyle{empty}}")
    tex_file.write_text("\n".join(lines), encoding="utf-8")
    compiled = False
    open(f"{tex_file}.bbl", "a").close()
    for compiler in ["pdflatex", "lualatex", "xelatex"]:
        try:
            subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error", str(tex_file)],
                cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30
            )
            if pdf_file.exists() and pdf_file.stat().st_size > 0:
                compiled = True; break
        except subprocess.SubprocessError:
            pass
    if not compiled: return None
    try:
        cropped_pdf = pdf_file.with_name(pdf_file.stem + "_cropped.pdf")
        # crop(["-c", "gb", "-p", "0", "-a", "-1", "-o", str(cropped_pdf), str(pdf_file)], quiet=True)
        crop_pdf_out_of_proc(str(pdf_file), str(cropped_pdf))
        if cropped_pdf.exists() and cropped_pdf.stat().st_size > 0:
            pdf_file = cropped_pdf
    except Exception:
        pass
    try:
        subprocess.run([PDFTOPPM, "-singlefile", "-png", str(pdf_file), str(pdf_file.with_suffix(""))], cwd=work_dir, check=True)
        png_path = pdf_file.with_suffix(".png")
        image = Image.open(png_path).convert("RGB")
        if image.getcolors(1) is not None:
            return None
        image = ImageOps.pad(image, (448, 448), color="white")
        image.save(png_path, format="PNG")
        return png_path
    except Exception:
        return None


_DETIKZIFY_SCORER = None
_CLIP_SCORER = None
_DREAMSIM_SCORER = None


def make_image_reward(work_dir, reward_model_dir, backend):
    png_cache = Path(work_dir) / "cache" / "tikz_png"
    png_cache.mkdir(parents=True, exist_ok=True)
    gt_feat_dir = Path(work_dir) / "cache" / "gt_feats"
    gt_feat_dir.mkdir(parents=True, exist_ok=True)

    global _DETIKZIFY_SCORER, _CLIP_SCORER, _DREAMSIM_SCORER

    if backend == "detikzify":
        if _DETIKZIFY_SCORER is None:
            _DETIKZIFY_SCORER = BatchedDeTikZifyScoreFast(
                reward_model_dir,
                batch_size=16,
                cache_size=4096,
                parity_mode=True,
                gt_feat_dir=gt_feat_dir,
            )
        scorer = _DETIKZIFY_SCORER
    
    elif backend == "clip":
        if _CLIP_SCORER is None:
            _CLIP_SCORER = BatchedClipScoreImgFast(
                reward_model_dir,
                batch_size=16,
                cache_size=4096,
                parity_mode=True,
                gt_feat_dir=gt_feat_dir,
            )
        scorer = _CLIP_SCORER
    
    elif backend == "dreamsim":
        if _DREAMSIM_SCORER is None:
            _DREAMSIM_SCORER = BatchedDreamSimScoreFast(
                model_name=reward_model_dir,
                cache_size=4096,
                cache_dir="models/dreamsim",
            )
        scorer = _DREAMSIM_SCORER

    else:
        raise ValueError(f"Unknown image_reward_backend: {backend}")

    def _tikz_key(tikz):
        return hashlib.sha1(tikz.encode("utf-8")).hexdigest()
    
    def _compile_or_load_png(tmp_dir, tikz):
        k = _tikz_key(tikz)
        cached = png_cache / f"{k}.png"
        if cached.exists():
            return str(cached)
        fig_id = f"tmp_{k[:10]}"
        out = process_figure_in_dir(tmp_dir, fig_id, tikz)
        if out is None:
            return None
        try:
            shutil.copy2(out, cached)
            return str(cached)
        except Exception:
            return str(out)
    
    MAX_WORKERS = max(2, min(8, mp.cpu_count() // 2))

    def image_similarity_reward(completions, ground_truth, reward_signal, **batch):
        tmp_root = Path(work_dir)
        tmp_root.mkdir(parents=True, exist_ok=True)
        batch_id = uuid.uuid4().hex[:8]
        batch_dir = tmp_root / f"batch_{batch_id}"
        (batch_dir / "comp").mkdir(parents=True, exist_ok=True)
        (batch_dir / "gt").mkdir(parents=True, exist_ok=True)

        comp_png_paths, gt_png_paths = [], []
        futures = []
        try:
            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                for comp in completions:
                    comp_doc = _extract_tikz_doc(comp)
                    if comp_doc is None:
                        futures.append(None)
                    else:
                        futures.append(ex.submit(_compile_or_load_png, batch_dir / "comp", comp_doc))
                for fut in futures:
                    if fut is None:
                        comp_png_paths.append(None)
                        continue
                    path = fut.result()
                    comp_png_paths.append(path)

            for i, _ in enumerate(completions):
                try:
                    gt_img = reward_signal[i]
                    if gt_img.mode != "RGB":
                        gt_img = gt_img.convert("RGB")
                    gt_img = ImageOps.pad(gt_img, (448, 448), color="white")
                    gt_path = batch_dir / "gt" / f"gt_{i}.png"
                    gt_img.save(gt_path, format="PNG")
                    gt_png_paths.append(str(gt_path))
                except Exception:
                    gt_png_paths.append(None)
            
            rewards, i1, i2, idxs = [], [], [], []
            for idx, (c, g) in enumerate(zip(comp_png_paths, gt_png_paths)):
                if c is None or g is None:
                    rewards.append(0.0)
                else:
                    i1.append(c)
                    i2.append(g)
                    idxs.append(idx)
                    rewards.append(None)
            if i1:
                scores = scorer(i1, i2)
                for k, sc in zip(idxs, scores):
                    rewards[k] = float(sc)
            total_rewards = [r if r is not None else 0.0 for r in rewards]
            return total_rewards
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    image_similarity_reward.__name__ = "image_similarity_reward"
    return image_similarity_reward


def _file_cache_key(path):
    try:
        st = os.stat(path)
        return f"{path}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        return f"{path}|-1|-1"


class BatchedDeTikZifyScoreFast:
    def __init__(self, model_name, device=None, batch_size=16, cache_size=4096, parity_mode=True, gt_feat_dir=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dtype = (torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32)
        model, processor = load_model(model_name_or_path=model_name, torch_dtype=base_dtype)
        self.vision = getattr(getattr(model, "model", model), "vision_model")
        if self.vision is None:
            raise RuntimeError("Could not find vision_model on reward model.")
        self.vision.to(self.device).eval()
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.proc = getattr(processor, "image_processor", processor)
        self.param_dtype = next(self.vision.parameters()).dtype
        self.batch_size = batch_size
        self._feat_cache = OrderedDict()
        self._cache_size = cache_size
        self._parity_mode = parity_mode
        self._gt_feat_dir = gt_feat_dir

    def _feat_disk_path(self, img_path):
        if self._gt_feat_dir is None:
            return None
        try:
            st = os.stat(img_path)
            fname = f"{Path(img_path).name}.{int(st.st_mtime)}.{st.st_size}.pt"
            return self._gt_feat_dir / fname
        except Exception:
            return None

    def _try_load_feat_from_disk(self, img_path):
        p = self._feat_disk_path(img_path)
        if p and p.exists():
            try:
                return torch.load(p, map_location="cpu")
            except Exception:
                return None
        return None

    def _pre_im(self, im):
        im = im.convert("RGB")
        return expand(im, max(im.size), do_trim=True)

    def _encode_images(self, pil_images):
        pil_images = [self._pre_im(im) for im in pil_images]
        enc = self.proc(images=pil_images, return_tensors="pt")
        px = enc["pixel_values"]
        if px.ndim == 3:
            px = px.unsqueeze(0)
        px = px.to(self.device, dtype=self.param_dtype, non_blocking=True)
        if self._parity_mode:
            with torch.inference_mode():
                out = self.vision(pixel_values=px)
                L = out.last_hidden_state.detach().to("cpu", dtype=torch.float32)
        else:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.device.type=="cuda", dtype=self.param_dtype):
                out = self.vision(pixel_values=px)
                L = out.last_hidden_state.detach().to("cpu", dtype=torch.float16)
        return [L[i] for i in range(L.shape[0])]

    def _get_feats_for_paths(self, paths):
        to_load_idx, to_load_imgs = [], []
        feats = [None] * len(paths)
        keys = [_file_cache_key(p) for p in paths]
        for i, (p, k) in enumerate(zip(paths, keys)):
            f = self._feat_cache.get(k, None)
            if f is not None:
                feats[i] = f
                continue
            f = self._try_load_feat_from_disk(p)
            if f is not None:
                feats[i] = f
                self._feat_cache[k] = f
                self._feat_cache.move_to_end(k)
                continue
            try:
                im = Image.open(p).convert("RGB"); im.load()
                to_load_idx.append(i); to_load_imgs.append(im)
            except Exception:
                feats[i] = None
        for start in range(0, len(to_load_imgs), self.batch_size):
            batch_imgs = to_load_imgs[start:start+self.batch_size]
            batch_idxs = to_load_idx[start:start+self.batch_size]
            encd = self._encode_images(batch_imgs)
            for i_local, f in enumerate(encd):
                i_global = batch_idxs[i_local]
                feats[i_global] = f
                k = keys[i_global]
                self._feat_cache[k] = f
                self._feat_cache.move_to_end(k)
                if len(self._feat_cache) > self._cache_size:
                    self._feat_cache.popitem(last=False)
                p_disk = self._feat_disk_path(paths[i_global])
                if p_disk is not None:
                    try:
                        torch.save(f, p_disk)
                    except Exception:
                        pass
        return feats

    @staticmethod
    def _emd_score(x_feats_cpu, y_feats_cpu):
        x = x_feats_cpu.to(torch.float32)
        y = y_feats_cpu.to(torch.float32)
        dists = 1.0 - pairwise_cosine_similarity(x.double(), y.double())
        emd_v = emd2(M=dists.numpy(), a=list(), b=list())
        score = 2 * np.tanh(-emd_v) + 1
        return float(max(0.0, min(1.0, score)))

    def __call__(self, image_paths1, image_paths2):
        assert len(image_paths1) == len(image_paths2)
        X = self._get_feats_for_paths(image_paths1)
        Y = self._get_feats_for_paths(image_paths2)
        scores = []
        for p1, p2, xf, yf in zip(image_paths1, image_paths2, X, Y):
            if xf is None or yf is None:
                scores.append(0.0); continue
            try:
                s = self._emd_score(xf, yf)
            except Exception as e:
                s = 0.0
            scores.append(s)
        return scores


class BatchedClipScoreImgFast:
    def __init__(self, model_name, device=None, batch_size=32, cache_size=4096, parity_mode=True, gt_feat_dir=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dtype = (torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32)
        self.param_dtype = base_dtype
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=base_dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self._feat_cache = OrderedDict()
        self._cache_size = cache_size
        self._parity_mode = parity_mode
        self._gt_feat_dir = Path(gt_feat_dir) if gt_feat_dir is not None else None

    def _feat_disk_path(self, img_path):
        if self._gt_feat_dir is None:
            return None
        try:
            st = os.stat(img_path)
            fname = f"{Path(img_path).name}.{int(st.st_mtime)}.{st.st_size}.pt"
            return self._gt_feat_dir / fname
        except Exception:
            return None

    def _try_load_feat_from_disk(self, img_path):
        p = self._feat_disk_path(img_path)
        if p and p.exists():
            try:
                return torch.load(p, map_location="cpu")
            except Exception:
                return None
        return None

    def _pre_im(self, im):
        im = im.convert("RGB")
        return expand(im, max(im.size), do_trim=True)

    def _encode_images(self, pil_images):
        pil_images = [self._pre_im(im) for im in pil_images]
        enc = self.proc(images=pil_images, return_tensors="pt")
        pixel_values = enc["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(self.device, dtype=self.param_dtype, non_blocking=True)
        if self._parity_mode:
            with torch.inference_mode():
                feats = self.model.get_image_features(pixel_values=pixel_values)
        else:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=self.param_dtype):
                feats = self.model.get_image_features(pixel_values=pixel_values)
        feats = feats.detach().to("cpu", dtype=torch.float32)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return [feats[i] for i in range(feats.shape[0])]

    def _get_feats_for_paths(self, paths):
        to_load_idx, to_load_imgs = [], []
        feats = [None] * len(paths)
        keys = [_file_cache_key(p) for p in paths]
        for i, (p, k) in enumerate(zip(paths, keys)):
            f = self._feat_cache.get(k, None)
            if f is not None:
                feats[i] = f
                continue
            f = self._try_load_feat_from_disk(p)
            if f is not None:
                feats[i] = f
                self._feat_cache[k] = f
                self._feat_cache.move_to_end(k)
                continue
            try:
                im = Image.open(p).convert("RGB")
                im.load()
                to_load_idx.append(i)
                to_load_imgs.append(im)
            except Exception:
                feats[i] = None
        for start in range(0, len(to_load_imgs), self.batch_size):
            batch_imgs = to_load_imgs[start : start + self.batch_size]
            batch_idxs = to_load_idx[start : start + self.batch_size]
            encd = self._encode_images(batch_imgs)
            for i_local, f in enumerate(encd):
                i_global = batch_idxs[i_local]
                feats[i_global] = f
                k = keys[i_global]
                self._feat_cache[k] = f
                self._feat_cache.move_to_end(k)
                if len(self._feat_cache) > self._cache_size:
                    self._feat_cache.popitem(last=False)
                p_disk = self._feat_disk_path(paths[i_global])
                if p_disk is not None:
                    try:
                        torch.save(f, p_disk)
                    except Exception:
                        pass
        return feats

    @staticmethod
    def _cos_score(x_feats_cpu, y_feats_cpu):
        x = x_feats_cpu.to(torch.float32).view(-1)
        y = y_feats_cpu.to(torch.float32).view(-1)
        x = x / x.norm().clamp_min(1e-6)
        y = y / y.norm().clamp_min(1e-6)
        cos = float((x * y).sum().item())
        s = (cos + 1.0) / 2.0
        return max(0.0, min(1.0, s))

    def __call__(self, image_paths1, image_paths2):
        assert len(image_paths1) == len(image_paths2)
        X = self._get_feats_for_paths(image_paths1)
        Y = self._get_feats_for_paths(image_paths2)
        scores = []
        for p1, p2, xf, yf in zip(image_paths1, image_paths2, X, Y):
            if xf is None or yf is None:
                scores.append(0.0)
                continue
            try:
                s = self._cos_score(xf, yf)
            except Exception as e:
                s = 0.0
            scores.append(s)
        return scores


class BatchedDreamSimScoreFast:
    def __init__(self, model_name="ensemble", device=None, dtype=None, cache_size=4096, cache_dir="models/dreamsim"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model, processor = dreamsim(dreamsim_type=model_name, pretrained=True, normalize_embeds=True, device=self.device, cache_dir=cache_dir)
        for extractor in model.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            extractor.proj = extractor.proj.to(self.dtype)
        self.model = model.to(self.device, self.dtype)
        self.processor = processor
        self._pair_cache = OrderedDict()
        self._cache_size = cache_size

    def _load_and_expand(self, path1, path2):
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        max_dim = max(img1.width, img1.height, img2.width, img2.height)
        bg = (255, 255, 255)
        img1 = ImageOps.pad(img1, (max_dim, max_dim), color=bg, centering=(0.5, 0.5))
        img2 = ImageOps.pad(img2, (max_dim, max_dim), color=bg, centering=(0.5, 0.5))
        return img1, img2

    def _pair_key(self, p1, p2):
        k1 = _file_cache_key(p1)
        k2 = _file_cache_key(p2)
        return f"{k1}||{k2}"

    def _score_pair(self, path1, path2):
        key = self._pair_key(path1, path2)
        cached = self._pair_cache.get(key, None)
        if cached is not None:
            self._pair_cache.move_to_end(key)
            return cached
        img1, img2 = self._load_and_expand(path1, path2)
        img1_tensor = self.processor(img1).to(self.device, self.dtype)
        img2_tensor = self.processor(img2).to(self.device, self.dtype)
        with torch.inference_mode():
            dist = self.model(img1_tensor, img2_tensor).item()
            sim = 1.0 - dist
        sim = float(max(0.0, min(1.0, sim)))
        self._pair_cache[key] = sim
        self._pair_cache.move_to_end(key)
        if len(self._pair_cache) > self._cache_size:
            self._pair_cache.popitem(last=False)
        return sim

    def __call__(self, image_paths1, image_paths2):
        assert len(image_paths1) == len(image_paths2)
        scores = []
        for p1, p2 in zip(image_paths1, image_paths2):
            if p1 is None or p2 is None:
                scores.append(0.0)
                continue
            try:
                s = self._score_pair(p1, p2)
            except Exception as e:
                s = 0.0
            scores.append(s)
        return scores


if __name__=="__main__":
    args = arg_parser()

    if args.image_reward_backend == "detikzify":
        reward_model_dir = f"{args.work_dir}/trained_models_detikzify/{args.reward_model_id}/{args.checkpoint_id_reward_model}"
    elif args.image_reward_backend == "clip":
        reward_model_dir = f"models/{args.reward_model_id}"
    elif args.image_reward_backend == "dreamsim":
        reward_model_dir = args.reward_model_id

    if args.base_model:
        cache_model_id = f"{args.model_type}_{args.max_seq_length}_{args.code_length[0]}_{args.code_length[1]}"
        grpo_model_id = f"{args.model_type}_{args.reward_model_id}_{args.checkpoint_id_reward_model}_{args.device_batch_size}_{args.gradient_accumulation_steps}_{args.num_generations}_{args.learning_rate}_{args.lr_scheduler_type}_{args.temperature}_{args.top_p}_{args.epsilon}_{args.weight_decay}_{args.epochs}_{args.loss_type}_{args.epsilon_high}"
        model_path = os.path.join(args.work_dir, "models", args.model_type)
    else:
        cache_model_id = f"{args.model_id}_{args.checkpoint_id}_{args.max_seq_length}_{args.code_length[0]}_{args.code_length[1]}"
        grpo_model_id = f"{args.model_type}_{args.checkpoint_id}_{args.reward_model_id}_{args.checkpoint_id_reward_model}_{args.device_batch_size}_{args.gradient_accumulation_steps}_{args.num_generations}_{args.learning_rate}_{args.lr_scheduler_type}_{args.temperature}_{args.top_p}_{args.epsilon}_{args.weight_decay}_{args.epochs}_{args.loss_type}_{args.epsilon_high}"
        model_path = os.path.join(args.work_dir, "trained_models_sft", args.model_id, args.checkpoint_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype="auto",
        # device_map="auto",
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if not args.base_model:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    save_dir = os.path.join(args.work_dir, "processed_dataset_grpo")
    shard_paths = sorted(
        [os.path.join(save_dir, d) for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))],
        key=lambda x: int(x.split("_")[-1])
    )
    if not shard_paths:
        raise FileNotFoundError(f"No .arrow files found in {save_dir}")
    datasets_list = [load_from_disk(p) for p in shard_paths]
    full_dataset = concatenate_datasets(datasets_list)

    tokenizer_path = os.path.join(args.work_dir, "tokenizer_grpo", f"{cache_model_id}.arrow")
    subset = full_dataset.shuffle(seed=args.seed)
    preprocessor = make_qwen_base_preprocessor_grpo(model_path, args.max_seq_length)
    processed_dataset = subset.map(
        preprocessor,
        batched=True,
        batch_size=256,
        num_proc=32,
        remove_columns=["text", "code", "image"],
        load_from_cache_file=True,
        cache_file_name=tokenizer_path,
        desc="Tokenizing",
    )

    if not args.base_model:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None:
            raise ValueError("Token <|im_end|> not found in tokenizer vocab.")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        gen_config = GenerationConfig(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id
        )
    
    else:
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_seq_length
        )

    trained_model_path = os.path.join(args.work_dir, "trained_models_grpo", grpo_model_id)
    grpo_config = GRPOConfig(
        output_dir=trained_model_path,
        # use_vllm=True,
        # vllm_mode="server",
        loss_type="dr_grpo",
        scale_rewards=False,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=int(args.max_seq_length / 4),
        max_completion_length=args.max_seq_length,
        num_generations=args.num_generations,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.device_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        # temperature=args.temperature,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        weight_decay=args.weight_decay,
        mask_truncated_completions=args.mask_truncated_completions,
		logging_dir=f"tf_logs_grpo/{grpo_model_id}",
        report_to="tensorboard",
        # logging_steps=5,
        save_steps=50,
        # eval_strategy="steps",
        # eval_steps=100,
        generation_kwargs=gen_config.to_dict(),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_image_reward(args.tmp_dir, reward_model_dir, args.image_reward_backend),
        args=grpo_config,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    last_checkpoint = None
    if os.path.isdir(trained_model_path):
        last_checkpoint = get_last_checkpoint(trained_model_path)
    trainer.train(resume_from_checkpoint=last_checkpoint)