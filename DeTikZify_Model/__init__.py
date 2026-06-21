from urllib.parse import urlparse

from datasets import DownloadManager
from safetensors.torch import load_file
from transformers.utils.hub import has_file
from transformers import AutoConfig, AutoProcessor, is_timm_available

try:
    from transformers import AutoModelForVision2Seq as _AutoVisionModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoVisionModel

from .configuration_detikzify import *
from .modeling_detikzify import *
from .processing_detikzify import *
from .adapter import load as load_adapter

if is_timm_available():
    from .v1 import models as v1_models, load as load_v1


def is_remote_url(url_or_filename):
    parsed = urlparse(str(url_or_filename))
    return parsed.scheme in ("http", "https")


def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        _AutoVisionModel.register(DetikzifyConfig, DetikzifyForConditionalGeneration)
        AutoProcessor.register(DetikzifyConfig, DetikzifyProcessor)
    except ValueError:
        pass  # already registered


def load(model_name_or_path, modality_projector=None, is_v1=False, **kwargs):
    if is_timm_available() and (is_v1 or model_name_or_path in v1_models):  # type: ignore
        model, tokenizer, image_processor = load_v1(  # type: ignore
            model_name_or_path=model_name_or_path,
            modality_projector=modality_projector,
            **kwargs
        )
        return model, DetikzifyProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_len=model.config.num_patches,
            image_token=tokenizer.convert_ids_to_tokens(model.config.patch_token_id)
        )

    register()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = _AutoVisionModel.from_pretrained(model_name_or_path, **kwargs)

    if modality_projector is not None:
        if is_remote_url(modality_projector):
            modality_projector = DownloadManager().download(modality_projector)
        model.load_state_dict(
            state_dict=load_file(
                filename=modality_projector,  # type: ignore
                device=str(model.device)
            ),
            strict=False
        )

    if has_file(model_name_or_path, "adapter/model.safetensors"):
        model, processor = load_adapter(model=model, processor=processor)

    return model, processor