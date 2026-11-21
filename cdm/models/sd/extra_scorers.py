"""
extra_scorers.py -- non-CLIP scorers for the final evaluation (eval_all_experiments.py).

The designs are optimized against CLIP, so the paper's MMD and the CLIP zero-shot gender
classifier are not independent of the optimization. These score the SAME generated portraits
with models that play no part in it:

  Embedder          a second image embedding (e.g. facebook/dinov2-base): MMD / SWD are then
                    computed between the methods' portraits and the target portraits in that space.
  GenderClassifier  a face-attribute classifier trained on FairFace (e.g.
                    dima806/fairface_gender_image_detection): p(male) from its predictions.

Both are Hugging Face transformers checkpoints, loaded at a pinned revision when one is given.
`python -m cdm.models.sd.extra_scorers --check <model_id> ...` prints what a checkpoint is (class,
labels, hidden size) without generating anything, so the label mapping can be checked first.
"""

import argparse

import torch
import torch.nn.functional as F

MALE_NAMES = {"male", "man", "men", "m"}
FEMALE_NAMES = {"female", "woman", "women", "f"}


class Embedder:
    """L2-normalised image embeddings: the pooled output (DINOv2: the layer-normed CLS token)."""

    def __init__(self, model_id, device, revision=None):
        from transformers import AutoImageProcessor, AutoModel
        self.model_id, self.device = model_id, device
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).eval()

    @torch.no_grad()
    def embed(self, images, batch=32):
        self.model.to(self.device)
        out = []
        for s in range(0, len(images), batch):
            inputs = self.processor(images=images[s:s + batch], return_tensors="pt").to(self.device)
            o = self.model(**inputs)
            e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            out.append(F.normalize(e.float(), dim=-1))
        self.model.to("cpu")
        return torch.cat(out, dim=0)


class GenderClassifier:
    """Image classifier with a male and a female label; p(male) per image."""

    def __init__(self, model_id, device, revision=None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.model_id, self.device = model_id, device
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision).eval()
        labels = {int(i): str(n).strip().lower() for i, n in self.model.config.id2label.items()}
        male = [i for i, n in labels.items() if n in MALE_NAMES]
        female = [i for i, n in labels.items() if n in FEMALE_NAMES]
        if len(male) != 1 or len(female) != 1:
            raise ValueError(f"{model_id}: cannot find one male and one female label in {labels}")
        self.male, self.female, self.labels = male[0], female[0], labels

    @torch.no_grad()
    def p_male(self, images, batch=32):
        """Per image, P(male) renormalised over the male and female classes."""
        self.model.to(self.device)
        out = []
        for s in range(0, len(images), batch):
            inputs = self.processor(images=images[s:s + batch], return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.float()
            two = torch.stack([logits[:, self.male], logits[:, self.female]], dim=1)
            out.append(two.softmax(dim=1)[:, 0].cpu())
        self.model.to("cpu")
        return torch.cat(out).tolist()


AGE_BRACKETS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")


class FairFace3Task:
    """The FairFace race / gender / age classifiers of wesjos/fairface-*-cnano-3task-224 (timm ConvNeXt-Nano,
    three linear heads on the 640-d pooled feature), rebuilt here from the repository's inference.py
    so that only the weights (model.pt) are loaded, never the repository's code. Gender logits are
    ordered (Male, Female); age is 8 ordinal thresholds, the bracket being the number of positive ones."""

    ARCH, FEATURE_DIM, SIZE = "convnext_nano.in12k_ft_in1k", 640, 224
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, model_id, device, revision=None):
        import os
        import timm
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        self.model_id, self.device = model_id, device
        if os.path.isdir(model_id):
            path = os.path.join(model_id, "model.pt")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_id, "model.pt", revision=revision)

        class Student(nn.Module):
            def __init__(s):
                super().__init__()
                s.backbone = timm.create_model(self.ARCH, pretrained=False, num_classes=0)
                s.race_head = nn.Linear(self.FEATURE_DIM, 7)
                s.gender_head = nn.Linear(self.FEATURE_DIM, 2)
                s.age_head = nn.Linear(self.FEATURE_DIM, 8)

            def forward(s, x):
                f = s.backbone(x)
                return {"gender": s.gender_head(f), "age": s.age_head(f)}

        try:   # tensors only, when the checkpoint allows it
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model = Student()
        self.model.load_state_dict(ckpt["ema"] if ckpt.get("selected_ema") else ckpt["model"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.SIZE, self.SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(mean=self.MEAN, std=self.STD)])

    @torch.no_grad()
    def predict(self, images, batch=64):
        """(p_male per image, age bracket index 0..8 per image)."""
        self.model.to(self.device)
        pm, age = [], []
        for s in range(0, len(images), batch):
            x = torch.stack([self.transform(im.convert("RGB")) for im in images[s:s + batch]]).to(self.device)
            out = self.model(x)
            pm += out["gender"].float().softmax(dim=1)[:, 0].cpu().tolist()
            age += (out["age"] > 0).sum(dim=1).cpu().tolist()
        self.model.to("cpu")
        return pm, age


def bracket_hist(brackets):
    """Fraction of images per FairFace age bracket (9 values)."""
    n = max(len(brackets), 1)
    return [sum(1 for b in brackets if b == i) / n for i in range(len(AGE_BRACKETS))]


def short_name(model_id):
    """Key suffix for the results: facebook/dinov2-base -> dinov2_base."""
    return model_id.split("/")[-1].replace("-", "_").replace(".", "_").lower()


def main(argv=None):
    p = argparse.ArgumentParser(description="print what Hugging Face image checkpoints are")
    p.add_argument("--check", nargs="+", required=True, metavar="MODEL_ID")
    a = p.parse_args(argv)
    from huggingface_hub import model_info
    from transformers import AutoConfig
    for m in a.check:
        info = model_info(m)
        cfg = AutoConfig.from_pretrained(m)
        card = info.card_data.to_dict() if info.card_data else {}
        print(f"== {m}\n  revision (commit): {info.sha}\n  architecture: {cfg.architectures}  model_type: {cfg.model_type}"
              f"\n  hidden size: {getattr(cfg, 'hidden_size', None)}\n  id2label: {getattr(cfg, 'id2label', None)}"
              f"\n  license: {card.get('license')}  datasets: {card.get('datasets')}  downloads: {info.downloads}")


if __name__ == "__main__":
    main()
