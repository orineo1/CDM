# Copied from orineo1/conditional-matching-paper @ cb8bc91 (branch claude/sd-shared-controlnet-hvt-uniform),
# path SD_cond_SD_controlnet/models.py
# SD-specific: the Stable Diffusion pipeline itself. Nothing here duplicates a shared package.

"""
models.py — Model loading for the MLGD-F pipeline.

Loads the Architect (SDXL Base) and Sprinter (SDXL Turbo + ControlNet-Scribble)
diffusion pipelines. Optionally loads a LoRA or fully fine-tuned U-Net onto the
Architect. Supports HuggingFace Hub paths via the `hf://` prefix.
"""

import logging
import os
import random
import warnings
import zipfile

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import hf_hub_download
from peft import PeftModel


# The pipeline casts the VAE between float16 and float32 several times per step, and
# every cast makes diffusers warn "There are modules in AutoencoderKL that should be
# kept in float32: []". The list is empty -- there is nothing to keep in float32 -- so
# the warning is noise (hundreds of lines per run); drop exactly that message, and
# only when the list is empty. The same casts trigger the `upcast_vae` deprecation
# notice, dropped for the same reason. Every other warning is left alone.
class _DropEmptyFp32Warning(logging.Filter):
    def filter(self, record):
        return "that should be kept in float32: []" not in record.getMessage()


def _quiet_vae_cast_warnings():
    diffusers_logging.get_logger("diffusers")        # make sure diffusers set up its handler
    root = logging.getLogger("diffusers")
    for h in root.handlers:                          # records from every diffusers module pass here
        h.addFilter(_DropEmptyFp32Warning())
    warnings.filterwarnings("ignore", message=r".*upcast_vae.*", category=FutureWarning)


_quiet_vae_cast_warnings()


def _resolve_lora_path(lora_path: str) -> str:
    """
    If lora_path starts with 'hf://', download from HuggingFace Hub.
    Expects format: hf://<repo_id>/<filename>
    If the downloaded file is a zip, extracts it and returns the folder path.
    Local paths pass through unchanged.
    """
    if not lora_path.startswith("hf://"):
        return lora_path

    without_prefix = lora_path[len("hf://"):]
    parts = without_prefix.split("/")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    if local_path.endswith(".zip"):
        extract_dir = local_path.replace(".zip", "")
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(extract_dir)

        if os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
            return extract_dir

        for sub in os.listdir(extract_dir):
            sub_path = os.path.join(extract_dir, sub)
            if os.path.isdir(sub_path):
                if os.path.exists(os.path.join(sub_path, "adapter_config.json")):
                    return sub_path

        raise FileNotFoundError(
            f"adapter_config.json not found inside {extract_dir}"
        )

    return local_path


# Single source of truth for the default ControlNet, so eval (notebooks calling
# load_models() with no override) and generation (run_mlgd_f.py's --controlnet_model_id
# default) can't silently drift onto different weights -- both import this constant
# instead of each hardcoding their own copy of the model id string.
DEFAULT_CONTROLNET_MODEL_ID = "xinsir/controlnet-scribble-sdxl-1.0"
DEFAULT_SPRINTER_MODEL_ID   = "stabilityai/sdxl-turbo"
DEFAULT_ARCHITECT_MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
HED_ANNOTATOR_ID            = "lllyasviel/Annotators"


def make_deterministic(seed):
    """
    Seed every RNG and force deterministic GPU kernels.

    Seeding alone makes a run deterministic *in order* (which noise is drawn), but the
    guidance backpropagates through the Sprinter, the VAE and CLIP at every step, and
    cuDNN convolution backward, flash / memory-efficient attention backward and some
    scatter ops pick non-deterministic kernels by default. Bit-level differences in the
    gradient grow over the guided steps into a different scribble. So: deterministic
    cuDNN, no autotuning, deterministic cuBLAS workspace, torch's deterministic-algorithm
    mode (warn_only: an op with no deterministic kernel prints a warning naming it
    instead of crashing) and the math attention kernel only. Slower, more memory in
    attention; same GPU model + same library versions + same seed -> same numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # seeds CPU and every CUDA device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # read at the first cuBLAS call
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            fn(False)
    torch.backends.cuda.enable_math_sdp(True)


def resolved_revision(repo_id, revision=None):
    """
    The commit sha of `repo_id` that from_pretrained loaded, read from the local
    HuggingFace cache (works offline, after the download). `revision=None` means the
    hub's `main`. Returns None if the cache has no entry (e.g. a local path).
    """
    if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    from huggingface_hub.constants import HF_HUB_CACHE
    ref = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"),
                       "refs", revision or "main")
    try:
        with open(ref) as f:
            return f.read().strip()
    except OSError:
        return None


def load_models(
    device,
    architect_lora_path=None,
    architect_unet_path=None,
    controlnet_model_id=DEFAULT_CONTROLNET_MODEL_ID,
    sprinter_model_id=DEFAULT_SPRINTER_MODEL_ID,
    architect_model_id=DEFAULT_ARCHITECT_MODEL_ID,
    controlnet_revision=None,
    sprinter_revision=None,
    architect_revision=None,
):
    """
    Load Architect and Sprinter pipelines.

    Args:
        device:               torch device string ('cuda' or 'cpu').
        architect_lora_path:  Optional LoRA path (local or hf://).
        architect_unet_path:  Optional fully fine-tuned U-Net path (takes
                              priority over architect_lora_path).
        controlnet_model_id:  HuggingFace model ID for the ControlNet.
        sprinter_model_id:    HuggingFace model ID for the Sprinter.
        architect_model_id:   HuggingFace model ID for the Architect.
        *_revision:           hub commit sha / tag to pin each model to (None = main;
                              the sha actually loaded is recorded by resolved_revision).

    Returns:
        (architect, sprinter) pipeline tuple.
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        revision=controlnet_revision,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    sprinter = StableDiffusionXLControlNetPipeline.from_pretrained(
        sprinter_model_id,
        revision=sprinter_revision,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    architect = StableDiffusionXLPipeline.from_pretrained(
        architect_model_id,
        revision=architect_revision,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    architect.scheduler = DDIMScheduler.from_config(architect.scheduler.config)

    if architect_unet_path:
        architect.unet = UNet2DConditionModel.from_pretrained(
            architect_unet_path, torch_dtype=torch.float16,
        ).to(device)
        print(f"Loaded fine-tuned architect U-Net from {architect_unet_path}")
    elif architect_lora_path:
        resolved_path = _resolve_lora_path(architect_lora_path)
        architect.unet = PeftModel.from_pretrained(architect.unet, resolved_path)
        print(f"Loaded architect LoRA from {resolved_path}")

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)
    architect.set_progress_bar_config(disable=True)
    sprinter.set_progress_bar_config(disable=True)

    original_call = StableDiffusionXLControlNetPipeline.__call__
    StableDiffusionXLControlNetPipeline.__call__ = lambda self, *args, **kwargs: (
        original_call.__wrapped__(self, *args, **kwargs)
        if hasattr(original_call, "__wrapped__")
        else original_call(self, *args, **kwargs)
    )

    return architect, sprinter


BASE_SCHEDULERS = {
    "euler": EulerDiscreteScheduler,          # SDXL Base's own default
    "dpm":   DPMSolverMultistepScheduler,     # DPM-Solver++(2M), good at 20-30 steps
    "ddim":  DDIMScheduler,
}


def build_base_controlnet_pipeline(architect, controlnet, scheduler="euler"):
    """
    The multi-step conditional sampler Y | X used to *evaluate* a scribble: SDXL Base
    (the Architect's weights) + the same ControlNet-Scribble as the Sprinter, sampled with
    a full schedule and CFG instead of SDXL Turbo's 2 steps. The optimisation never sees
    this sampler, so a target matched under it is a check that the scribble transfers
    beyond the few-step model it was optimised against.

    Shares the Architect's modules (no second copy of the weights); the scheduler is a
    fresh instance so the Architect's DDIM state is not touched.
    """
    sched = BASE_SCHEDULERS[scheduler].from_config(architect.scheduler.config)
    pipe = StableDiffusionXLControlNetPipeline(
        vae=architect.vae,
        text_encoder=architect.text_encoder,
        text_encoder_2=architect.text_encoder_2,
        tokenizer=architect.tokenizer,
        tokenizer_2=architect.tokenizer_2,
        unet=architect.unet,
        controlnet=controlnet,
        scheduler=sched,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def freeze_module(module):
    """Freeze all parameters in a module (no gradient updates)."""
    for p in module.parameters():
        p.requires_grad_(False)


def setup_gradient_checkpointing(architect, sprinter):
    """
    Enable gradient checkpointing on both pipelines and freeze all weights.
    Must be called before the MLGD-F loop.
    """
    architect.unet.enable_gradient_checkpointing()
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    for m in [
        architect.unet,
        architect.vae,
        sprinter.unet,
        sprinter.controlnet,
        sprinter.vae,
    ]:
        freeze_module(m)
