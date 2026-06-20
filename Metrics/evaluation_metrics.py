import re
import torch
import pickle
import lpips as lpips_lib
import torch.nn.functional as F

from torch import nn
from hashlib import md5
from pathlib import Path
from dreamsim import dreamsim
from collections import Counter
from sacrebleu import corpus_bleu
from torchvision import transforms
from pygments.lexers import TexLexer
from sacremoses import MosesTokenizer
from functools import cached_property
from PIL import Image, ImageOps, ImageChops
from pygments.token import Comment, Text, Name
from torchmetrics.text import ExtendedEditDistance
from transformers import AutoProcessor, AutoModel, ViTImageProcessor, ViTModel

from detikzify.model import load as load_model
from detikzify.evaluate.imagesim import ImageSim


class FeatureWrapper(nn.Module):
    def __init__(self, model_name, device, dtype):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    @cached_property
    def model(self):
        model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        return model.to(self.device)

    def forward(self, pixel_values):
        with torch.inference_mode():
            return self.model.get_image_features(pixel_values.to(self.device, self.dtype))


def trim(image, bg="white"):
    bg_img = Image.new(image.mode, image.size, bg)
    diff = ImageChops.difference(image, bg_img)
    return image.crop(diff.getbbox()) if diff.getbbox() else image


def expand(image, size, do_trim=False, bg="white"):
    if do_trim:
        image = trim(image, bg=bg)
    return ImageOps.pad(image, (size, size), color=bg, method=Image.Resampling.LANCZOS)


def remove_alpha(image, bg):
    bg_img = Image.new("RGBA", image.size, bg)
    return Image.alpha_composite(bg_img, image.convert("RGBA")).convert("RGB")


def load(image, bg="white", timeout=None):
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    return remove_alpha(image, bg=bg)


class ClipScore:
    def __init__(self, model_name, device=None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image_path, description):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=description, return_tensors="pt")
        inputs = {k: v.to(self.device) if v.dtype != torch.float32 else v.to(self.device, self.dtype) for k, v in inputs.items()}
        with torch.inference_mode():
            image_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_embeds = self.model.get_text_features(input_ids=inputs["input_ids"])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds @ text_embeds.T).squeeze().item()
        return similarity


class BatchedClipScore:
    def __init__(self, model_name, device=None, dtype=None, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image_paths, descriptions):
        assert len(image_paths) == len(descriptions)
        scores = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_imgs  = [Image.open(p).convert("RGB") for p in image_paths[i:i + self.batch_size]]
            batch_texts = descriptions[i:i + self.batch_size]
            max_len = getattr(self.processor.tokenizer, "model_max_length", 64)
            inputs = self.processor(
                images=batch_imgs,
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            inputs = {
                k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                img_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                txt_embeds = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
                batch_scores = (img_embeds * txt_embeds).sum(dim=-1).tolist()
                scores.extend(batch_scores)
        return scores


class ClipScoreImg:
    def __init__(self, model_name, device=None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image_path1, image_path2):
        img1 = Image.open(image_path1).convert("RGB")
        img2 = Image.open(image_path2).convert("RGB")
        inputs = self.processor(images=[img1, img2], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            image_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds[0] @ image_embeds[1]).item()
        return similarity


class BatchedClipScoreImg:
    def __init__(self, model_name, device=None, dtype=None, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image_paths1, image_paths2):
        assert len(image_paths1) == len(image_paths2)
        scores = []
        for i in range(0, len(image_paths1), self.batch_size):
            batch_img1 = image_paths1[i:i + self.batch_size]
            batch_img2 = image_paths2[i:i + self.batch_size]
            imgs1 = [Image.open(p).convert("RGB") for p in batch_img1]
            imgs2 = [Image.open(p).convert("RGB") for p in batch_img2]
            inputs = self.processor(images=imgs1 + imgs2, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                embeds = self.model.get_image_features(pixel_values=pixel_values)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            embeds1 = embeds[:len(batch_img1)]
            embeds2 = embeds[len(batch_img1):]
            sims = (embeds1 * embeds2).sum(dim=-1).tolist()
            scores.extend(sims)
        return scores


class DeTikZifyScore:
    def __init__(self, model_name, device=None, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        model, processor = load_model(model_name_or_path=model_name, device_map=device, torch_dtype=dtype)
        self.imagesim = ImageSim.from_detikzify(model=model, processor=processor, mode="emd")

    def __call__(self, image_path1, image_path2):
        score = self.imagesim.get_similarity(img1=image_path1, img2=image_path2)
        return score


class BatchedDeTikZifyScore:
    def __init__(self, model_name, device=None, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        model, processor = load_model(model_name_or_path=model_name, device_map=device, torch_dtype=dtype)
        self.imagesim = ImageSim.from_detikzify(model=model, processor=processor, mode="emd")

    def __call__(self, image_paths1, image_paths2):
        assert len(image_paths1) == len(image_paths2)
        scores = []
        iterable = zip(image_paths1, image_paths2)
        for img1, img2 in iterable:
            score = self.imagesim.get_similarity(img1=img1, img2=img2)
            scores.append(score)
        return scores


class DinoScore:
    def __init__(self, model_name_or_path="models/dino-vits16", device=None, use_cls_token=True, normalize=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cls_token = use_cls_token
        self.normalize = normalize
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        if self.use_cls_token:
            feats = hidden[:, 0]
        else:
            feats = hidden.mean(dim=1)
        if self.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0)

    @torch.no_grad()
    def __call__(self, pred_path, gt_path):
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)
        f_pred = self._encode(img_pred)
        f_gt = self._encode(img_gt)
        sim = F.cosine_similarity(f_pred, f_gt, dim=0).item()
        return float(sim)


class LPIPS:
    def __init__(self, net="alex", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips_lib.LPIPS(net=net).to(self.device)
        self.to_tensor = transforms.ToTensor()

    def _preprocess(self, image, size=None):
        if image.mode != "RGB":
            image = image.convert("RGB")
        if size is not None:
            image = image.resize(size, Image.BICUBIC)
        t = self.to_tensor(image).unsqueeze(0)
        t = t * 2.0 - 1.0
        return t.to(self.device)

    @torch.no_grad()
    def __call__(self, pred_path, gt_path):
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)
        target_size = img_gt.size
        x = self._preprocess(img_pred, size=target_size)
        y = self._preprocess(img_gt, size=target_size)
        d = self.loss_fn(x, y)
        return float(d.mean().item())


class CrystalBLEU:
    SUPPORTED_ENVIRONMENTS = ["tikzpicture", "circuitikz", "tikzcd"]
    def __init__(self, corpus=None, k=500, n=4, use_cache=True, cache_dir="models/crystalbleu", cache_key=None, only_code=False):
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.k = k
        self.n = n
        self.use_cache = use_cache
        self.only_code = only_code
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trivially_shared_ngrams = self._load_or_compute_trivial_ngrams(corpus, cache_key)

    def _extract_relevant_code(self, tex):
        env_pattern = rf"\\begin\{{({'|'.join(self.SUPPORTED_ENVIRONMENTS)})\}}(?:\[[^\]]*\])?"
        pattern = re.compile(env_pattern + r".*?\\end\{\1\}", re.DOTALL)
        match = pattern.search(tex)
        if match:
            content = match.group(0)
        else:
            content = tex
        content = re.sub(r"\s+", " ", content).strip()
        return content

    def _get_hashname(self, corpus, cache_key):
        base = cache_key if cache_key else "".join(sorted(corpus))
        prefix = "onlycode_" if self.only_code else "full_"
        return prefix + md5(base.encode()).hexdigest()

    def _load_or_compute_trivial_ngrams(self, corpus, cache_key):
        if not corpus and not cache_key:
            raise ValueError("Either a corpus or a cache_key must be provided.")
        hashname = self._get_hashname(corpus, cache_key)
        cache_path = self.cache_dir / f"{hashname}.pkl"
        if cache_path.exists() and self.use_cache:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        if not corpus:
            raise ValueError(f"Cache file not found at {cache_path} and no corpus provided to generate it.")
        all_ngrams = []
        for tex in corpus:
            tokens = self._tokenize(tex)
            for n in range(1, self.n + 1):
                all_ngrams.extend(self._ngrams(tokens, n))
        frequencies = Counter(all_ngrams)
        shared = dict(frequencies.most_common(self.k))
        if self.use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(shared, f)
        return shared

    def _tokenize(self, text):
        if self.only_code:
            text = self._extract_relevant_code(text)
        tokens = []
        for tok_type, val in self.lexer.get_tokens(text):
            if val.strip() and tok_type != Comment:
                if tok_type in {Text, Name.Attribute, Name.Builtin}:
                    tokens.extend(self.tokenizer.tokenize(val.strip()))
                else:
                    tokens.append(val.strip())
        return tokens

    def _ngrams(self, tokens, n):
        return list(zip(*[tokens[i:] for i in range(n)]))

    def _filter_ngrams(self, tokens):
        filtered = []
        for n in range(1, self.n + 1):
            for ng in self._ngrams(tokens, n):
                if ng not in self.trivially_shared_ngrams:
                    filtered.append(" ".join(ng))
        return filtered

    def __call__(self, predictions, references):
        assert len(predictions) == len(references)
        hyps = []
        refs = []
        for hyp, ref in zip(predictions, references):
            hyp_tok = self._tokenize(hyp)
            ref_tok = self._tokenize(ref)
            hyp_filtered = self._filter_ngrams(hyp_tok)
            ref_filtered = self._filter_ngrams(ref_tok)
            if hyp_filtered and ref_filtered:
                hyps.append(" ".join(hyp_filtered))
                refs.append([" ".join(ref_filtered)])
        if not hyps or not refs:
            return 0.0
        bleu = corpus_bleu(hyps, refs)
        return bleu.score / 100.0


class DreamSim:
    def __init__(self, model_name="ensemble", pretrained=True, normalize=True, dtype=None, device=None, cache_dir="models/dreamsim"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        model, processor = dreamsim(
            dreamsim_type=model_name,
            pretrained=pretrained,
            normalize_embeds=normalize,
            device=self.device,
            cache_dir=cache_dir)
        for extractor in model.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            extractor.proj = extractor.proj.to(self.dtype)
        self.model = model.to(self.device, self.dtype)
        self.processor = processor

    def _load_and_expand(self, path1, path2):
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        max_dim = max(img1.width, img1.height, img2.width, img2.height)
        bg = (255, 255, 255)
        img1 = ImageOps.pad(img1, (max_dim, max_dim), color=bg, centering=(0.5, 0.5))
        img2 = ImageOps.pad(img2, (max_dim, max_dim), color=bg, centering=(0.5, 0.5))
        return img1, img2

    def __call__(self, img1_path, img2_path):
        img1, img2 = self._load_and_expand(img1_path, img2_path)
        img1_tensor = self.processor(img1).to(self.device, self.dtype)
        img2_tensor = self.processor(img2).to(self.device, self.dtype)
        with torch.inference_mode():
            dist = self.model(img1_tensor, img2_tensor).item()
            sim = 1.0 - dist
        return sim


class TexEditDistance:
    SUPPORTED_ENVIRONMENTS = ["tikzpicture", "circuitikz", "tikzcd"]
    def __init__(self, only_code, alpha=0.5, rho=1.5, deletion=1.0, insertion=1.0):
        self.only_code = only_code
        self.alpha = alpha
        self.rho = rho
        self.deletion = deletion
        self.insertion = insertion
        self.lexer = TexLexer()
        self.eed = ExtendedEditDistance(alpha=alpha, rho=rho, deletion=deletion, insertion=insertion, return_sentence_level_score=False)

    def _extract_relevant_code(self, tex):
        env_pattern = rf"\\begin\{{({'|'.join(self.SUPPORTED_ENVIRONMENTS)})\}}(?:\[[^\]]*\])?"
        pattern = re.compile(env_pattern + r".*?\\end\{\1\}", re.DOTALL)
        match = pattern.search(tex)
        if match:
            content = match.group(0)
        else:
            content = tex
        content = re.sub(r"\s+", " ", content).strip()
        return content

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def _tokenize(self, text):
        tokens = []
        for tok_type, value in self.lexer.get_tokens(text):
            if value.strip() and tok_type is not Comment:
                if tok_type is Text:
                    tokens.extend(self._preprocess_text(value).split())
                else:
                    tokens.extend(value.strip().split())
        return " ".join(tokens)

    def __call__(self, hyp, ref):
        if self.only_code:
            hyp = self._extract_relevant_code(hyp)
            ref = self._extract_relevant_code(ref)
        hyp_proc = self._tokenize(hyp)
        ref_proc = self._tokenize(ref)
        return self.eed([hyp_proc], [[ref_proc]]).item()