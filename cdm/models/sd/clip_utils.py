# Copied from orineo1/conditional-matching-paper @ cb8bc91 (branch claude/sd-shared-controlnet-hvt-uniform),
# path SD_cond_SD_controlnet/clip_utils.py
# SD-specific: the Stable Diffusion pipeline itself. Nothing here duplicates a shared package.

"""
clip_utils.py — CLIP model loading and differentiable image encoding.

Used to encode Sprinter-generated images into 768-dim embeddings for
MMD/SWD loss computation in the MLGD-F guidance loop.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def load_clip_model(device, model_id=CLIP_MODEL_ID, revision=None):
    """
    Load and freeze a CLIP ViT-L/14 model. `revision` pins the hub commit (None = main).

    Returns:
        (clip_model, clip_processor)
    """
    clip_processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
    clip_model = CLIPModel.from_pretrained(model_id, revision=revision).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    return clip_model, clip_processor


def encode_images_clip(pixel_tensor, clip_model, clip_processor):
    """
    Encode a batch of images through CLIP's vision encoder.

    Differentiable w.r.t. pixel_tensor — gradients flow back through
    resize, normalize, and the CLIP ViT forward pass.

    Args:
        pixel_tensor: [B, 3, H, W] float tensor with values in [0, 1].
        clip_model:   frozen CLIPModel.
        clip_processor: CLIPProcessor (mean/std constants only).

    Returns:
        [B, 768] L2-normalised CLIP embeddings.
    """
    # Differentiable resize to 224x224
    resized = F.interpolate(
        pixel_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )

    # CLIP normalisation constants
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=pixel_tensor.device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=pixel_tensor.device
    ).view(1, 3, 1, 1)
    normalized = (resized - mean) / std

    emb = clip_model.vision_model(pixel_values=normalized).pooler_output
    emb = clip_model.visual_projection(emb)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # [B, 768]
