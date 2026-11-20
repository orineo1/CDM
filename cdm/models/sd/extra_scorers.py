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
