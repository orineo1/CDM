# Port of orineo1/conditional-matching-paper SD_cond_SD_controlnet/notebooks/eval_all_experiments.ipynb
# Entry point. Run as:  python -m cdm.models.sd.eval_all_experiments --help
#                  or:  bash experiments/stable_diffusion/four_targets/run.sh eval <scenario>

"""
eval_all_experiments.py -- the paper's final evaluation (MMD / SWD / p(male) tables), N = 2,000.

For one scenario:
    1. Build the target distribution, N_EVAL images generated from the SOURCE scribble with
       the group prompts and fractions (age mode: N_EVAL // n_ages per age), CLIP-encoded.
    2. For every method's scribble found (source, avg_prompt, sdedit_best, mlgd_f; any
       other one with --scribbles name=path): generate N_EVAL portraits with the neutral prompt, CLIP-encode, and
       compute MMD and SWD against the target, plus p(male) with a binomial 95% CI in the
       gender scenarios. Delta% = improvement over the source scribble.

Same protocol, seeds and prompts as the notebook, so its numbers are reproducible; runs
under the deterministic settings and the pinned model revisions of run_mlgd_f.

Saved to --output_dir: eval_results.json, eval_table.md, embeddings.npz (target + every
method, so the metrics can be recomputed without a GPU), preview_<method>.png and
target_preview_<group>.png (5 images each), and the three plots. The 2,000 photos per
method are NOT saved. Logged to wandb: the table, previews, plots and every number.

Scribbles are read from --run_dir (a run_mlgd_f output directory, or one of the paper's
experiments/<Scenario>/ directories), and --scribbles name=path adds or overrides one.
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cdm.models.sd.clip_utils import CLIP_MODEL_ID, encode_images_clip, load_clip_model
from cdm.models.sd.extra_scorers import Embedder, GenderClassifier, short_name
from cdm.models.sd.generation import generate_and_store_cs
from cdm.models.sd.metrics import compute_mmd, compute_swd
from cdm.models.sd.models import (
    DEFAULT_ARCHITECT_MODEL_ID,
    DEFAULT_CONTROLNET_MODEL_ID,
    DEFAULT_SPRINTER_MODEL_ID,
    HED_ANNOTATOR_ID,
    load_models,
    make_deterministic,
    resolved_revision,
)
from cdm.models.sd.run_mlgd_f import compute_clip_softmax, library_versions
from cdm.models.sd.visualization import plot_row

NEUTRAL_PROMPT = "a superrealistic professional photograph of"
MAN_PROMPT     = "a superrealistic portrait photograph of a man, studio lighting"
WOMAN_PROMPT   = "a superrealistic portrait photograph of a woman, studio lighting"
N_PREVIEW      = 5

# method name -> file names looked for in --run_dir: run_mlgd_f's own names first, then the
# names eval_baselines.py / the paper's experiments/<Scenario>/ use. Order = table order.
SCRIBBLE_FILES = {   # the methods looked for in --run_dir; anything else: --scribbles name=path
    "source":      ["scribble.png", "scribble_source.png"],
    "avg_prompt":  ["baselines/scribble_avg_prompt.png", "scribble_avg_prompt.png"],
    "sdedit_best": ["baselines/scribble_sdedit_best.png", "scribble_sdedit_best.png"],
    "mlgd_f":      ["final_scribble_mlgd_f.png", "scribble_mlgdd.png"],
}
METHOD_COLORS = {"source": "#888888", "avg_prompt": "#17becf", "sdedit_best": "#4C72B0", "mlgd_f": "#2ca02c"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, default=None,
                   help="directory holding the scribbles (see SCRIBBLE_FILES in this file)")
    p.add_argument("--scribbles", type=str, nargs="*", default=[], metavar="NAME=PATH",
                   help="extra scribbles, or overrides of the ones found in --run_dir")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_eval", type=int, default=2000, help="target size and images per method")
    p.add_argument("--seed", type=int, default=42, help="the notebook's seed")
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--eval_prompt", type=str, default=NEUTRAL_PROMPT)

    p.add_argument("--mode", type=str, default="gender", choices=["gender", "age"])
    p.add_argument("--target_prompts", type=str, nargs="+", default=None, metavar="NAME:PROMPT:N",
                   help="gender mode: group specs as in run_mlgd_f; N sets the fraction N/sum")
    p.add_argument("--age_min", type=int, default=40)
    p.add_argument("--age_max", type=int, default=79, help="inclusive, as in the notebook")
    p.add_argument("--age_step", type=int, default=1)
    p.add_argument("--age_gender", type=str, default="man")

    p.add_argument("--controlnet_model_id", type=str, default=DEFAULT_CONTROLNET_MODEL_ID)
    p.add_argument("--sprinter_model_id",   type=str, default=DEFAULT_SPRINTER_MODEL_ID)
    p.add_argument("--architect_model_id",  type=str, default=DEFAULT_ARCHITECT_MODEL_ID)
    p.add_argument("--controlnet_revision", type=str, default=None)
    p.add_argument("--sprinter_revision",   type=str, default=None)
    p.add_argument("--architect_revision",  type=str, default=None)
    p.add_argument("--hed_revision",        type=str, default=None, help="unused here; accepted so run.sh can pass it")
    p.add_argument("--clip_revision",       type=str, default=None)

    g = p.add_argument_group("non-CLIP scorers (cdm.models.sd.extra_scorers): the same portraits, scored again")
    g.add_argument("--extra_embedder", type=str, default=None, metavar="MODEL_ID",
                   help="a second image embedding for MMD / SWD, e.g. facebook/dinov2-base")
    g.add_argument("--extra_embedder_revision", type=str, default=None)
    g.add_argument("--gender_classifier", type=str, default=None, metavar="MODEL_ID",
                   help="a face-attribute classifier for p(male), e.g. dima806/fairface_gender_image_detection")
    g.add_argument("--gender_classifier_revision", type=str, default=None)

    p.add_argument("--wandb_project", type=str, default="sdxl-eval")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Pure helpers (no GPU): also used by the tests and export
# ---------------------------------------------------------------------------

def parse_groups(target_prompts):
    """'name:prompt:n' specs -> [(name, prompt, frac)], fractions normalised to 1."""
    if not target_prompts:
        target_prompts = [f"Man:{MAN_PROMPT}:50", f"Woman:{WOMAN_PROMPT}:50"]
    groups = []
    for spec in target_prompts:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"--target_prompts entry '{spec}' must be 'name:prompt:n'")
        groups.append((parts[0].strip(), parts[1].strip(), float(parts[2])))
    total = sum(g[2] for g in groups)
    return [(n, p, c / total) for n, p, c in groups]


def find_scribbles(run_dir, overrides):
    """{method: path} for every scribble present, in table order. Overrides win."""
    found = {}
    if run_dir:
        for name, candidates in SCRIBBLE_FILES.items():
            for c in candidates:
                path = os.path.join(run_dir, c)
                if os.path.exists(path):
                    found[name] = path
                    break
    for spec in overrides:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--scribbles entry '{spec}' must be NAME=PATH")
        found[name.strip()] = path
    order = list(SCRIBBLE_FILES) + [n for n in found if n not in SCRIBBLE_FILES]
    return {n: found[n] for n in order if n in found}


def binomial_ci(n_success, n_total, z=1.96):
    p = n_success / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    return p, p - z * se, p + z * se


def add_deltas(results):
    """Improvement over the source scribble, absolute and in %, in place."""
    if "source" not in results:
        return results
    s_mmd, s_swd = results["source"]["mmd"], results["source"]["swd"]
    for r in results.values():
        r["mmd_improvement"] = s_mmd - r["mmd"]
        r["mmd_improvement_pct"] = (s_mmd - r["mmd"]) / s_mmd * 100
        r["swd_improvement"] = s_swd - r["swd"]
        r["swd_improvement_pct"] = (s_swd - r["swd"]) / s_swd * 100
    return results


def results_table(results, scenario, n_eval, gender):
    """Markdown table in the shape of the paper's results tables."""
    head = "| Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) |" + (" p(male) | 95% CI | n_male / n_female |" if gender else "")
    sep = "|---|---|---|---|---|" + ("---|---|---|" if gender else "")
    lines = [f"# {scenario}: N = {n_eval}", "", head, sep]
    for name, r in results.items():
        row = (f"| {name} | {r['mmd']:.3f} | {r.get('mmd_improvement_pct', 0):+.1f}% | "
               f"{r['swd']:.4f} | {r.get('swd_improvement_pct', 0):+.1f}% |")
        if gender:
            row += (f" {r['p_male']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |"
                    f" {r['n_male']} / {r['n_female']} |")
        lines.append(row)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plots (matplotlib only)
# ---------------------------------------------------------------------------

def bar_plot(results, key, ylabel, title, path):
    names = list(results)
    vals = [results[n][key] for n in names]
    colors = [METHOD_COLORS.get(n, "#4C72B0") for n in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, vals, color=colors, alpha=0.85)
    if "source" in results:
        src = results["source"][key]
        ax.axhline(src, color="gray", linestyle="--", linewidth=1.0, label="source")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{(src - v) / src * 100:+.1f}%", ha="center", va="bottom", fontsize=9)
        ax.legend()
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def ci_plot(results, target_frac, title, path):
    names = list(results)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, n in enumerate(names):
        r = results[n]
        ax.errorbar(i, r["p_male"], yerr=[[r["p_male"] - r["ci_lo"]], [r["ci_hi"] - r["p_male"]]],
                    fmt="o", capsize=6, linewidth=1.5, color=METHOD_COLORS.get(n, "#4C72B0"), markersize=8)
    ax.axhline(target_frac, color="gray", linestyle="--", linewidth=0.8, label=f"target p(male)={target_frac:.2f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("p(male)"); ax.set_ylim(0, 1); ax.set_title(title); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# GPU part
# ---------------------------------------------------------------------------

def gen_images(sprinter, scribble, prompt, n, cn_scale, seed):
    """n Sprinter portraits of `scribble`, own generator (the notebook's gen_images)."""
    gen = torch.Generator(device=sprinter.device).manual_seed(seed)
    imgs, _ = generate_and_store_cs(sprinter, prompt, scribble, n, batch_size=2,
                                    cn_scale=cn_scale, generator=gen)
    return imgs


def clip_embed(images, clip_model, clip_processor, device, batch=32):
    clip_model.to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            t = torch.cat([TF.to_tensor(im).unsqueeze(0) for im in images[s:s + batch]]).to(device)
            out.append(encode_images_clip(t, clip_model, clip_processor))
    clip_model.to("cpu")
    return torch.cat(out, dim=0)


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    if args.no_deterministic:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    else:
        make_deterministic(args.seed)
    print(f"Device: {device}  seed: {args.seed}  N_EVAL: {args.n_eval}", flush=True)

    scribbles = find_scribbles(args.run_dir, args.scribbles)
    if not scribbles:
        sys.exit(f"no scribbles found in {args.run_dir!r}; expected one of "
                 + ", ".join(c for cs in SCRIBBLE_FILES.values() for c in cs))
    if "source" not in scribbles:
        sys.exit("a 'source' scribble is required: the target is built from it and deltas are relative to it")
    missing = [m for m in SCRIBBLE_FILES if m not in scribbles]
    print("Scribbles: " + ", ".join(f"{k}={v}" for k, v in scribbles.items()), flush=True)
    if missing:
        print(f"Not found (skipped): {', '.join(missing)}", flush=True)
    scribble_imgs = {k: Image.open(v).convert("RGB") for k, v in scribbles.items()}
    scenario = os.path.basename(os.path.normpath(args.output_dir))

    architect, sprinter = load_models(
        device, controlnet_model_id=args.controlnet_model_id, sprinter_model_id=args.sprinter_model_id,
        architect_model_id=args.architect_model_id, controlnet_revision=args.controlnet_revision,
        sprinter_revision=args.sprinter_revision, architect_revision=args.architect_revision)
    del architect   # only the Sprinter generates here
    torch.cuda.empty_cache()
    clip_model, clip_processor = load_clip_model(device, revision=args.clip_revision)
    model_revisions = {
        args.sprinter_model_id:   resolved_revision(args.sprinter_model_id, args.sprinter_revision),
        args.controlnet_model_id: resolved_revision(args.controlnet_model_id, args.controlnet_revision),
        CLIP_MODEL_ID:            resolved_revision(CLIP_MODEL_ID, args.clip_revision),
    }
    emb2 = clf2 = None
    if args.extra_embedder:
        emb2 = Embedder(args.extra_embedder, device, revision=args.extra_embedder_revision)
        k2 = short_name(args.extra_embedder)
        model_revisions[args.extra_embedder] = resolved_revision(args.extra_embedder, args.extra_embedder_revision)
    if args.gender_classifier and args.mode == "gender":
        clf2 = GenderClassifier(args.gender_classifier, device, revision=args.gender_classifier_revision)
        kc = short_name(args.gender_classifier)
        model_revisions[args.gender_classifier] = resolved_revision(args.gender_classifier, args.gender_classifier_revision)
        print(f"{args.gender_classifier}: labels {clf2.labels} (male={clf2.male}, female={clf2.female})", flush=True)
    print("Models loaded.", flush=True)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                         config={**vars(args), "scribbles": scribbles, "model_revisions": model_revisions,
                                 "versions": library_versions()})
        print(f"wandb run: {run.name}", flush=True)

    # ── 1. target distribution, from the source scribble ────────────────────
    t0 = time.time()
    gender = args.mode == "gender"
    source = scribble_imgs["source"]
    target_embs, target_groups, embeddings = [], [], {}
    target_embs2 = []
    if gender:
        groups = parse_groups(args.target_prompts)
        for i, (name, prompt, frac) in enumerate(groups):
            n_i = max(1, int(args.n_eval * frac))
            print(f"Target [{name}] n={n_i} ...", flush=True)
            imgs = gen_images(sprinter, source, prompt, n_i, args.controlnet_scale, args.seed + i * 1000)
            e = clip_embed(imgs, clip_model, clip_processor, device)
            target_embs.append(e); embeddings[f"target_{name}"] = e.cpu().numpy()
            if emb2:
                e2 = emb2.embed(imgs); target_embs2.append(e2); embeddings[f"{k2}_target_{name}"] = e2.cpu().numpy()
            target_groups.append((name, n_i))
            plot_row(imgs, f"Target: {name}", count=N_PREVIEW,
                     save_path=os.path.join(args.output_dir, f"target_preview_{name}.png"))
            del imgs
        target_frac = groups[0][2]
    else:
        ages = list(range(args.age_min, args.age_max + 1, args.age_step))
        n_per_age = max(1, args.n_eval // len(ages))
        anchors = {ages[0], ages[len(ages) // 4], ages[len(ages) // 2], ages[3 * len(ages) // 4], ages[-1]}
        print(f"Target: {len(ages)} ages x {n_per_age} ...", flush=True)
        for age in ages:
            prompt = (f"a superrealistic portrait photograph of a {age}-year-old {args.age_gender}, "
                      "studio lighting, sharp focus, photographic")
            imgs = gen_images(sprinter, source, prompt, n_per_age, args.controlnet_scale, args.seed + age * 7)
            e = clip_embed(imgs, clip_model, clip_processor, device)
            target_embs.append(e); embeddings[f"target_age{age}"] = e.cpu().numpy()
            if emb2:
                e2 = emb2.embed(imgs); target_embs2.append(e2); embeddings[f"{k2}_target_age{age}"] = e2.cpu().numpy()
            target_groups.append((str(age), n_per_age))
            if age in anchors:
                plot_row(imgs, f"Target: age {age}", count=N_PREVIEW,
                         save_path=os.path.join(args.output_dir, f"target_preview_age{age}.png"))
            del imgs
        target_frac = None
    target_clip = torch.cat(target_embs, dim=0)
    target2 = torch.cat(target_embs2, dim=0).to(target_clip.device) if emb2 else None
    print(f"Target CLIP: {tuple(target_clip.shape)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # ── 2. every method ─────────────────────────────────────────────────────
    results = {}
    for name, scr in scribble_imgs.items():
        t1 = time.time()
        print(f"[{name}] generating {args.n_eval} images ...", flush=True)
        imgs = gen_images(sprinter, scr, args.eval_prompt, args.n_eval, args.controlnet_scale, args.seed)
        embs = clip_embed(imgs, clip_model, clip_processor, device)
        embeddings[f"method_{name}"] = embs.cpu().numpy()
        with torch.no_grad():
            mmd = compute_mmd(embs, target_clip).item()          # already sqrt(|MMD^2| + eps)
            # SWD: the notebook's adaptive estimator (projections added until it moves by
            # < 1e-3). The notebook drew the projections from the unseeded global RNG, so its
            # SWD differed run to run; here they come from a generator seeded with --seed.
            swd = compute_swd(embs, target_clip, seed=args.seed).item()
        r = {"mmd": mmd, "swd": swd, "n": args.n_eval}
        if gender:   # the notebook's classifier, unchanged: CLIP zero-shot man vs woman
            sr, _ = compute_clip_softmax(imgs, clip_model, clip_processor, MAN_PROMPT, WOMAN_PROMPT, device)
            n_male = sum(1 for x in sr if x["label"] == "male")
            p, lo, hi = binomial_ci(n_male, args.n_eval)
            r.update(p_male=p, ci_lo=lo, ci_hi=hi, n_male=n_male, n_female=args.n_eval - n_male)
            print(f"    MMD={mmd:.5f}  SWD={swd:.5f}  p(male)={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]", flush=True)
        else:
            print(f"    MMD={mmd:.5f}  SWD={swd:.5f}", flush=True)
        if emb2:   # the same portraits in the second embedding
            e2 = emb2.embed(imgs).to(target2.device)
            embeddings[f"{k2}_method_{name}"] = e2.cpu().numpy()
            with torch.no_grad():
                r[f"mmd_{k2}"] = compute_mmd(e2, target2).item()
                r[f"swd_{k2}"] = compute_swd(e2, target2, seed=args.seed).item()
            print(f"    {args.extra_embedder}: MMD={r[f'mmd_{k2}']:.5f}  SWD={r[f'swd_{k2}']:.5f}", flush=True)
        if clf2:   # p(male) from the face-attribute classifier, and how often it agrees with CLIP's label
            pm = clf2.p_male(imgs)
            n2 = sum(1 for x in pm if x > 0.5)
            p2, lo2, hi2 = binomial_ci(n2, args.n_eval)
            agree = sum(1 for x, c in zip(pm, sr) if (x > 0.5) == (c["label"] == "male")) / args.n_eval
            r.update({f"p_male_{kc}": p2, f"ci_lo_{kc}": lo2, f"ci_hi_{kc}": hi2, f"n_male_{kc}": n2,
                      f"agreement_with_clip_{kc}": agree})
            print(f"    {args.gender_classifier}: p(male)={p2:.3f}  95%CI=[{lo2:.3f},{hi2:.3f}]  "
                  f"agrees with CLIP on {agree:.1%}", flush=True)
        r["seconds"] = time.time() - t1
        results[name] = r
        plot_row(imgs, f"{name}  (MMD={mmd:.4f})", count=N_PREVIEW,
                 save_path=os.path.join(args.output_dir, f"preview_{name}.png"))
        del imgs, embs
        torch.cuda.empty_cache()
    add_deltas(results)
    if emb2 and "source" in results:
        for loss in ("mmd", "swd"):
            s0 = results["source"][f"{loss}_{k2}"]
            for r in results.values():
                r[f"{loss}_{k2}_improvement_pct"] = (s0 - r[f"{loss}_{k2}"]) / s0 * 100

    # ── 3. save ─────────────────────────────────────────────────────────────
    np.savez_compressed(os.path.join(args.output_dir, "embeddings.npz"), **embeddings)
    table = results_table(results, scenario, args.n_eval, gender)
    with open(os.path.join(args.output_dir, "eval_table.md"), "w") as f:
        f.write(table)
    if emb2 or clf2:
        f2 = [f"# {scenario}: non-CLIP scores of the same portraits (N = {args.n_eval})", ""]
        head, sep = "| Method | CLIP MMD |", "|---|---|"
        if emb2:
            head += f" {k2} MMD ↓ | Δ (%) | {k2} SWD ↓ | Δ (%) |"; sep += "---|---|---|---|"
        if clf2:
            head += f" CLIP p(male) | {kc} p(male) | 95% CI | agreement |"; sep += "---|---|---|---|"
        f2 += [head, sep]
        for n, r in results.items():
            row = f"| {n} | {r['mmd']:.3f} |"
            if emb2:
                row += (f" {r[f'mmd_{k2}']:.3f} | {r.get(f'mmd_{k2}_improvement_pct', 0):+.1f}% |"
                        f" {r[f'swd_{k2}']:.4f} | {r.get(f'swd_{k2}_improvement_pct', 0):+.1f}% |")
            if clf2:
                row += (f" {r['p_male']:.3f} | {r[f'p_male_{kc}']:.3f} | [{r[f'ci_lo_{kc}']:.3f}, {r[f'ci_hi_{kc}']:.3f}] |"
                        f" {r[f'agreement_with_clip_{kc}']:.1%} |")
            f2.append(row)
        with open(os.path.join(args.output_dir, "eval_table_nonclip.md"), "w") as f:
            f.write("\n".join(f2) + "\n")
        print("\n" + "\n".join(f2), flush=True)
    bar_plot(results, "mmd", "MMD", f"{scenario}: MMD vs target (N={args.n_eval})",
             os.path.join(args.output_dir, "mmd_bar.png"))
    bar_plot(results, "swd", "SWD", f"{scenario}: SWD vs target (N={args.n_eval})",
             os.path.join(args.output_dir, "swd_bar.png"))
    if gender:
        ci_plot(results, target_frac, f"{scenario}: p(male) 95% CI (N={args.n_eval})",
                os.path.join(args.output_dir, "p_male_ci.png"))
    record = {
        "scenario": scenario, "mode": args.mode, "n_eval": args.n_eval, "seed": args.seed,
        "deterministic": not args.no_deterministic, "eval_prompt": args.eval_prompt,
        "controlnet_scale": args.controlnet_scale,
        "target_groups": target_groups, "target_n": int(target_clip.shape[0]),
        "scribbles": scribbles, "results": results,
        "model_revisions": model_revisions, "versions": library_versions(),
        "total_minutes": (time.time() - t0) / 60,
        "wandb": ({"project": run.project, "entity": run.entity, "name": run.name, "id": run.id, "url": run.url}
                  if run else None),
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(record, f, indent=2)
    print("\n" + table, flush=True)

    if run:
        import wandb
        cols = ["method", "mmd", "mmd_improvement_pct", "swd", "swd_improvement_pct"] + \
               (["p_male", "ci_lo", "ci_hi", "n_male", "n_female"] if gender else [])
        wtable = wandb.Table(columns=cols, data=[[n] + [results[n].get(c) for c in cols[1:]] for n in results])
        media = {f"preview/{n}": wandb.Image(os.path.join(args.output_dir, f"preview_{n}.png")) for n in results}
        media.update({f"target/{g}": wandb.Image(os.path.join(args.output_dir, f"target_preview_{g}.png"))
                      for g, _ in target_groups
                      if os.path.exists(os.path.join(args.output_dir, f"target_preview_{g}.png"))})
        for plot in ("mmd_bar", "swd_bar", "p_male_ci"):
            pth = os.path.join(args.output_dir, f"{plot}.png")
            if os.path.exists(pth):
                media[f"plot/{plot}"] = wandb.Image(pth)
        wandb.log({"results": wtable, **media})
        for n, r in results.items():
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    wandb.summary[f"{n}/{k}"] = v
        wandb.finish()
    print(f"All outputs saved to {args.output_dir}  ({record['total_minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
