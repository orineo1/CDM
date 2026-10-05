# Entry point. Run as:  python -m cdm.models.sd.eval_baselines --help

"""
eval_baselines.py -- build the baseline scribbles of one scenario and save them.

    sdedit_best   the best of N unguided SDEdit candidates of the source scribble, each
                  scored by MMD on n_eval_search (250) portraits against a search target
                  built from the source scribble; N is set so the search takes as long as
                  the MLGD-F run did (its optimization_time_sec in metrics.json)
    avg_prompt    the HED scribble of ONE portrait generated from the frac-weighted average
                  of the groups' CLIP text embeddings (prompt interpolation)
    avg, sdedit   the older pixel-average scribble and one guided SDEdit, with --methods

Reads the run's scribble.png / final_scribble_mlgd_f.png (or the paper's
scribble_source.png / scribble_mlgdd.png) from --run_dir and writes to <run_dir>/baselines:
scribble_<method>.png, preview_<name>.png (n_preview portraits of every scribble, source and
MLGD-F included, under the neutral prompt), target_<group>.png, scribbles_all.png,
sdedit_search.png, baselines_meta.json. Logs the same to wandb.
`run.sh eval` then finds the scribbles there.

    bash experiments/stable_diffusion/four_targets/run.sh baselines balanced
    python -m cdm.models.sd.eval_baselines --run_dir <run> --scenario BalancedTarget
"""

import os, sys, copy, argparse, json, time
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── SDEdit settings per scenario ──────────────────────────────────────────────
# Only what the SDEdit baselines need and config/<scenario>.env does not carry. The
# target prompts / fractions, seed, steps and ControlNet scale come from the command line
# (run.sh passes the run's config), so the baselines always use the run's own settings.

NEUTRAL_PROMPT = 'a superrealistic professional photograph of'
SDEDIT = {
    'SkewedTarget':        dict(sdedit_prompt='a face scribble sketch, mostly feminine features sometimes with masculine features'),
    'BalancedTarget':      dict(sdedit_prompt='a face scribble sketch of man or woman'),
    'GenderInterpolation': dict(sdedit_prompt='a face scribble sketch, with a range of feminine to masculine features'),
    'AgeInterpolation':    dict(sdedit_prompt='a face scribble sketch of a man between 40 and 79 years old'),
}
SDEDIT_CFG    = 7.5    # guidance scale of the single guided SDEdit (--methods sdedit)
N_EVAL_SEARCH = 250    # portraits per candidate in the sdedit_best search


def build_cfg(args):
    """The dict the builders below read, from the command line."""
    cfg = dict(seed=args.seed, n_steps=args.n_steps, sdedit_start=args.start_step,
               controlnet_scale=args.controlnet_scale, neutral_prompt=args.eval_prompt,
               n_eval_search=args.n_eval_search, sdedit_cfg=SDEDIT_CFG,
               sdedit_prompt=SDEDIT[args.scenario]['sdedit_prompt'])
    if args.mode == 'age':
        cfg.update(mode='age', age_min=args.age_min, age_max=args.age_max + 1,   # builders use range(min, max)
                   age_step=args.age_step, n_per_age=args.n_per_age, age_gender=args.age_gender)
    else:
        groups = []
        for spec in args.target_prompts or [f'Man:{MAN_PROMPT}:50', f'Woman:{WOMAN_PROMPT}:50']:
            name, prompt, n = spec.split(':', 2)
            groups.append(dict(label=name.strip(), prompt=prompt.strip(), n=float(n)))
        total = sum(g['n'] for g in groups)
        cfg.update(mode='binary' if len(groups) == 2 else 'multiclass',
                   groups=[dict(label=g['label'], prompt=g['prompt'], frac=g['n'] / total) for g in groups])
    return cfg


MAN_PROMPT   = 'a superrealistic portrait photograph of a man, studio lighting'
WOMAN_PROMPT = 'a superrealistic portrait photograph of a woman, studio lighting'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def gen_images(sprinter, scribble_pil, prompt, n, controlnet_scale, seed=None):
    sprinter.vae.to(dtype=torch.float16)
    generator = torch.Generator(device=sprinter.device).manual_seed(seed) if seed is not None else None
    imgs = []
    with torch.no_grad():
        for start in range(0, n, 2):
            bs = min(2, n - start)
            imgs.extend(sprinter(
                prompt=[prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
                output_type='pil',
                generator=generator,
            ).images)
    sprinter.vae.to(dtype=torch.float32)
    return imgs


def clip_embed(images, clip_model, clip_processor, device):
    from cdm.models.sd.clip_utils import encode_images_clip
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in images]).to(device)
    clip_model.to(device)
    with torch.no_grad():
        embs = encode_images_clip(tensors, clip_model, clip_processor)
    clip_model.to('cpu')
    return embs


def pil_to_tensor(pil_list, device):
    return torch.cat([TF.to_tensor(img).unsqueeze(0) for img in pil_list], dim=0).to(device)


# ── SDEdit ────────────────────────────────────────────────────────────────────

def run_sdedit(scr_pil, seed, architect, device, cfg_scale, n_steps_total, sdedit_start, prompt=''):
    architect.vae.to(dtype=torch.float32)
    architect.scheduler.set_timesteps(n_steps_total, device=device)
    timesteps = architect.scheduler.timesteps

    with torch.no_grad():
        t_in   = TF.to_tensor(scr_pil.convert('RGB')).unsqueeze(0).to(device).float()
        latent = architect.vae.encode((t_in * 2) - 1).latent_dist.mean * architect.vae.config.scaling_factor

    alpha   = architect.scheduler.alphas_cumprod.to(device)[timesteps[sdedit_start].long()].float()
    noise   = torch.randn(latent.shape,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          device=device, dtype=torch.float32)
    latents = ((alpha ** 0.5) * latent + ((1 - alpha) ** 0.5) * noise).half()
    sched   = copy.deepcopy(architect.scheduler)

    if cfg_scale > 0:
        with torch.no_grad():
            pe, ne, pooled, neg_pooled = architect.encode_prompt(
                prompt=prompt, negative_prompt='', device=device,
                do_classifier_free_guidance=True, num_images_per_prompt=1)
        add_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype, device=device)
        added   = {'text_embeds': torch.cat([neg_pooled, pooled]), 'time_ids': add_ids.repeat(2, 1)}
        cfg_st  = torch.cat([ne, pe])
        with torch.no_grad():
            for t in timesteps[sdedit_start:]:
                lmi     = sched.scale_model_input(torch.cat([latents] * 2), t)
                out     = architect.unet(lmi, t, encoder_hidden_states=cfg_st,
                                         added_cond_kwargs=added, return_dict=False)[0]
                nu, nc  = out.chunk(2)
                latents = sched.step(nu + cfg_scale * (nc - nu), t, latents).prev_sample
    else:
        with torch.no_grad():
            pe, _, pooled, _ = architect.encode_prompt(
                prompt='', negative_prompt='', device=device,
                do_classifier_free_guidance=False, num_images_per_prompt=1)
        add_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype, device=device)
        added   = {'text_embeds': pooled, 'time_ids': add_ids}
        with torch.no_grad():
            for t in timesteps[sdedit_start:]:
                lmi     = sched.scale_model_input(latents, t)
                out     = architect.unet(lmi, t, encoder_hidden_states=pe,
                                         added_cond_kwargs=added, return_dict=False)[0]
                latents = sched.step(out, t, latents).prev_sample

    with torch.no_grad():
        dec = architect.vae.decode(
            (latents.float() / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
    return T.ToPILImage()(torch.clamp((dec.float() + 1) / 2, 0, 1).squeeze(0).cpu())


# ── BUILD TARGET EMBEDDINGS ───────────────────────────────────────────────────

def build_target_embeddings(cfg, source_scribble, sprinter, clip_model, clip_processor, device):
    from cdm.models.sd.clip_utils import encode_images_clip
    SEED     = cfg['seed']
    cn_scale = cfg['controlnet_scale']
    mode     = cfg['mode']
    group_imgs = {}

    if mode in ('binary', 'multiclass'):
        n_target = cfg['n_eval_search']
        all_imgs = []
        for i, g in enumerate(cfg['groups']):
            n_i  = max(1, int(n_target * g['frac']))
            print(f"  [{g['label']}] n={n_i}...")
            imgs = gen_images(sprinter, source_scribble, g['prompt'], n_i, cn_scale, seed=SEED + i * 1000)
            all_imgs.extend(imgs)
            group_imgs[g['label']] = imgs
        target_clip = clip_embed(all_imgs, clip_model, clip_processor, device)

    elif mode == 'age':
        ages      = list(range(cfg['age_min'], cfg['age_max'], cfg['age_step']))
        n_per_age = cfg['n_per_age']
        age_embs  = {}
        clip_model.to(device)
        with torch.no_grad():
            for age in tqdm(ages, desc='Building age target'):
                prompt = (f"a superrealistic portrait photograph of a {age}-year-old {cfg.get('age_gender', 'man')}, "
                          'studio lighting, sharp focus, photographic')
                imgs = gen_images(sprinter, source_scribble, prompt, n_per_age, cn_scale, seed=SEED + age * 7)
                embs = encode_images_clip(pil_to_tensor(imgs, device), clip_model, clip_processor)
                age_embs[age] = embs.cpu()
                group_imgs[str(age)] = imgs
        clip_model.to('cpu')
        target_clip = torch.cat([age_embs[a] for a in ages], dim=0).to(device)

    return target_clip, group_imgs


# ── AVG SCRIBBLE ──────────────────────────────────────────────────────────────

def build_avg_scribble(cfg, source_scribble, sprinter, hed, device):
    SEED     = cfg['seed']
    cn_scale = cfg['controlnet_scale']
    mode     = cfg['mode']

    if mode == 'age':
        ages = list(range(cfg['age_min'], cfg['age_max'], cfg['age_step']))
        prompts = [
            f"a superrealistic portrait photograph of a {age}-year-old {cfg.get('age_gender', 'man')}, studio lighting, sharp focus, photographic"
            for age in ages]
        fracs = [1.0 / len(ages)] * len(ages)
    else:
        prompts = [g['prompt'] for g in cfg['groups']]
        fracs   = [g['frac']   for g in cfg['groups']]

    sprinter.vae.to(dtype=torch.float16)
    hed_scribbles = []
    for i, (prompt, frac) in enumerate(zip(prompts, fracs)):
        gen = torch.Generator(device=sprinter.device).manual_seed(SEED + i)
        with torch.no_grad():
            portrait = sprinter(
                prompt=[prompt], image=[source_scribble],
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=cn_scale,
                output_type='pil', generator=gen,
            ).images[0]
        hed_scribbles.append(np.array(hed(portrait, scribble=True)).astype(np.float32))
    sprinter.vae.to(dtype=torch.float32)

    avg_np = sum(f * s for f, s in zip(fracs, hed_scribbles)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg_np)


# ── AVG-PROMPT SCRIBBLE (CLIP text-embedding interpolation) ─────────────────────

def build_avg_scribble_prompt_interp(cfg, source_scribble, sprinter, hed, device):
    """
    Interpolates in CLIP TEXT-PROMPT EMBEDDING space: encode each group's prompt once,
    take the frac-weighted average of (prompt_embeds, pooled_prompt_embeds) and generate a
    SINGLE portrait from that one averaged embedding -- unlike build_avg_scribble, which
    generates one portrait PER group and blends their HED scribbles in pixel space.
    Returns the HED scribble of that portrait.
    """
    SEED     = cfg['seed']
    cn_scale = cfg['controlnet_scale']
    mode     = cfg['mode']

    if mode == 'age':
        ages = list(range(cfg['age_min'], cfg['age_max'], cfg['age_step']))
        prompts = [
            f"a superrealistic portrait photograph of a {age}-year-old {cfg.get('age_gender', 'man')}, studio lighting, sharp focus, photographic"
            for age in ages]
        fracs = [1.0 / len(ages)] * len(ages)
    else:
        prompts = [g['prompt'] for g in cfg['groups']]
        fracs   = [g['frac']   for g in cfg['groups']]

    avg_embeds, avg_pooled = None, None
    for prompt, frac in zip(prompts, fracs):
        with torch.no_grad():
            prompt_embeds, _, pooled_embeds, _ = sprinter.encode_prompt(
                prompt=prompt, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        avg_embeds = frac * prompt_embeds if avg_embeds is None else avg_embeds + frac * prompt_embeds
        avg_pooled = frac * pooled_embeds if avg_pooled is None else avg_pooled + frac * pooled_embeds

    gen = torch.Generator(device=sprinter.device).manual_seed(SEED)
    sprinter.vae.to(dtype=torch.float16)
    with torch.no_grad():
        portrait = sprinter(
            prompt_embeds=avg_embeds, pooled_prompt_embeds=avg_pooled,
            image=[source_scribble],
            num_inference_steps=2, guidance_scale=0.0,
            controlnet_conditioning_scale=cn_scale,
            output_type='pil', generator=gen,
        ).images[0]
    sprinter.vae.to(dtype=torch.float32)

    scribble_np = np.array(hed(portrait, scribble=True)).astype(np.uint8)
    return Image.fromarray(scribble_np), portrait


# ── SAVE HELPERS ──────────────────────────────────────────────────────────────

def save_scribble_grid(scribbles_dict, out_dir):
    names  = list(scribbles_dict.keys())
    images = list(scribbles_dict.values())
    n      = len(names)

    for name, img in zip(names, images):
        img.save(out_dir / f'scribble_{name}.png')
        print(f'  Saved scribble_{name}.png')

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, img, name in zip(axes, images, names):
        ax.imshow(img, cmap='gray'); ax.set_title(name); ax.axis('off')
    plt.suptitle('All Scribbles', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'scribbles_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved scribbles_all.png')


def save_target_previews(group_imgs, out_dir, n_preview=5):
    for label, imgs in group_imgs.items():
        safe = label.replace(' ', '_').replace('/', '_')
        n_cols = min(n_preview, len(imgs))
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
        if n_cols == 1: axes = [axes]
        for ax, img in zip(axes, imgs[:n_cols]):
            ax.imshow(img); ax.axis('off')
        fig.suptitle(f'Target: {label}', fontweight='bold')
        plt.tight_layout()
        fig.savefig(out_dir / f'target_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved target_{safe}.png')


def save_sdedit_search_plot(candidate_mmds, best_idx, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(candidate_mmds, color='steelblue', alpha=0.7, linewidth=1)
    ax.scatter([best_idx], [candidate_mmds[best_idx]], color='red', zorder=5,
               label=f'Best (i={best_idx}, MMD={candidate_mmds[best_idx]:.5f})')
    ax.set_xlabel('Candidate index'); ax.set_ylabel('MMD (search)')
    ax.set_title('SDEdit unguided candidate search')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'sdedit_search.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved sdedit_search.png')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def _first(d, *names):
    for n in names:
        if (d / n).exists():
            return d / n
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run_dir',        type=str, required=True,
                        help='the MLGD-F run directory (outputs/full_<Scenario>_seed<S>_<id>), or a '
                             "directory with the paper's scribble_source.png / scribble_mlgdd.png")
    parser.add_argument('--scenario',       type=str, default=None, choices=list(SDEDIT),
                        help='default: from the run directory name (full_<Scenario>[_seed<S>_<id>])')
    parser.add_argument('--methods',        type=str, nargs='+', default=['sdedit_best', 'avg_prompt'],
                        choices=['sdedit_best', 'avg_prompt', 'avg', 'sdedit'])
    parser.add_argument('--output_dir',     type=str, default=None, help='default: <run_dir>/baselines')
    parser.add_argument('--lgd_cm_minutes', type=float, default=None,
                        help='the SDEdit search budget in minutes; default: the run\'s '
                             'optimization_time_sec from metrics.json')
    parser.add_argument('--max_candidates', type=int, default=None,
                        help='cap on the SDEdit candidates (for quick checks)')
    parser.add_argument('--n_eval_search',  type=int, default=N_EVAL_SEARCH,
                        help='portraits per SDEdit candidate for its MMD score')
    parser.add_argument('--n_preview',      type=int, default=5)
    parser.add_argument('--no_deterministic', action='store_true')
    # the run's own settings (run.sh passes config/<scenario>.env; a run_mlgd_f flag set on
    # the run should be repeated here)
    parser.add_argument('--seed',             type=int,   default=1)
    parser.add_argument('--n_steps',          type=int,   default=250)
    parser.add_argument('--start_step',       type=int,   default=125)
    parser.add_argument('--controlnet_scale', type=float, default=0.5)
    parser.add_argument('--eval_prompt',      type=str,   default=NEUTRAL_PROMPT)
    parser.add_argument('--mode',             type=str,   default='gender', choices=['gender', 'age'])
    parser.add_argument('--target_prompts',   type=str,   nargs='+', default=None, metavar='NAME:PROMPT:N')
    parser.add_argument('--age_min',          type=int,   default=40)
    parser.add_argument('--age_max',          type=int,   default=79, help='inclusive')
    parser.add_argument('--age_step',         type=int,   default=1)
    parser.add_argument('--n_per_age',        type=int,   default=3,
                        help='images per age in the sdedit_best search target')
    parser.add_argument('--age_gender',       type=str,   default='man')
    for m in ('architect', 'sprinter', 'controlnet', 'hed', 'clip'):
        parser.add_argument(f'--{m}_revision', type=str, default=None)
    parser.add_argument('--wandb_project',  type=str, default='sdxl-baselines')
    parser.add_argument('--wandb_entity',   type=str, default='',
                        help='wandb entity/team (leave empty to use logged-in user)')
    args = parser.parse_args()

    from cdm.models.sd.models import make_deterministic, load_models, resolved_revision
    from cdm.models.sd.clip_utils import load_clip_model, CLIP_MODEL_ID
    from cdm.models.sd.hed import load_hed, HED_ANNOTATOR_ID
    from cdm.models.sd.visualization import plot_row

    run_dir = Path(args.run_dir).resolve()
    scenario = args.scenario
    if scenario is None:
        base = run_dir.name.split('_', 1)[-1].split('_seed', 1)[0]
        scenario = base if base in SDEDIT else None
    if scenario is None:
        sys.exit(f'cannot tell the scenario from {run_dir.name}; pass --scenario ({", ".join(SDEDIT)})')
    args.scenario = scenario
    cfg  = build_cfg(args)
    SEED = cfg['seed']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_deterministic:
        torch.manual_seed(SEED); np.random.seed(SEED)
    else:
        make_deterministic(SEED)

    source_path = _first(run_dir, 'scribble.png', 'scribble_source.png')
    mlgd_path   = _first(run_dir, 'final_scribble_mlgd_f.png', 'scribble_mlgdd.png')
    if source_path is None:
        sys.exit(f'no source scribble (scribble.png / scribble_source.png) in {run_dir}')
    source_scribble = Image.open(source_path).convert('RGB')
    mlgd_scribble   = Image.open(mlgd_path).convert('RGB') if mlgd_path else None

    lgd_minutes = args.lgd_cm_minutes
    if 'sdedit_best' in args.methods and lgd_minutes is None:
        mpath = run_dir / 'metrics.json'
        if mpath.exists():
            lgd_minutes = json.load(open(mpath)).get('optimization_time_sec', 0) / 60
        if not lgd_minutes:
            sys.exit(f'no optimization_time_sec in {mpath}; pass --lgd_cm_minutes')
    print(f'\n=== {scenario} | {run_dir} -> {out_dir} | methods: {", ".join(args.methods)} | seed {SEED} ===')
    if lgd_minutes:
        print(f'SDEdit search budget: {lgd_minutes:.1f} min (the MLGD-F run)')

    # ── W&B init ──
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,   # None = use whoever is logged in
        config=dict(scenario=scenario, run_dir=str(run_dir), methods=args.methods,
                    lgd_cm_minutes=lgd_minutes, max_candidates=args.max_candidates,
                    deterministic=not args.no_deterministic,
                    **{k: v for k, v in cfg.items() if k != 'groups'}),
    )

    # ── Load models ──
    architect, sprinter = load_models(device, architect_revision=args.architect_revision,
                                      sprinter_revision=args.sprinter_revision,
                                      controlnet_revision=args.controlnet_revision)
    clip_model, clip_processor = load_clip_model(device, revision=args.clip_revision)
    hed = load_hed(args.hed_revision)
    model_revisions = {
        'architect': resolved_revision('stabilityai/stable-diffusion-xl-base-1.0', args.architect_revision),
        'sprinter': resolved_revision('stabilityai/sdxl-turbo', args.sprinter_revision),
        HED_ANNOTATOR_ID: resolved_revision(HED_ANNOTATOR_ID, args.hed_revision),
        CLIP_MODEL_ID: resolved_revision(CLIP_MODEL_ID, args.clip_revision),
    }
    print('Models loaded.')

    scribbles = {}          # method -> PIL, what gets saved
    meta = dict(scenario=scenario, run_dir=str(run_dir), methods=args.methods, seed=SEED,
                source=str(source_path), mlgd_f=str(mlgd_path) if mlgd_path else None,
                model_revisions=model_revisions)

    # ── avg_prompt: prompt-embedding interpolation ──
    if 'avg_prompt' in args.methods:
        print('\nBuilding avg_prompt scribble (CLIP text-embedding interpolation)...')
        scribbles['avg_prompt'], portrait = build_avg_scribble_prompt_interp(
            cfg, source_scribble, sprinter, hed, device)
        portrait.save(out_dir / 'avg_prompt_portrait.png')

    # ── avg (pixel average of per-group HED scribbles), sdedit (one guided SDEdit) ──
    if 'avg' in args.methods:
        print('\nBuilding avg scribble...')
        scribbles['avg'] = build_avg_scribble(cfg, source_scribble, sprinter, hed, device)
    if 'sdedit' in args.methods:
        print('\nBuilding guided SDEdit scribble...')
        scribbles['sdedit'] = run_sdedit(
            source_scribble, seed=SEED, architect=architect, device=device,
            cfg_scale=cfg['sdedit_cfg'], n_steps_total=cfg['n_steps'],
            sdedit_start=cfg['sdedit_start'], prompt=cfg['sdedit_prompt'])

    # ── sdedit_best: time-matched search, MMD on n_eval_search portraits ──
    if 'sdedit_best' in args.methods:
        from cdm.models.sd.metrics import compute_mmd, mmd_report
        print('\nBuilding search target...')
        target_clip, group_imgs = build_target_embeddings(
            cfg, source_scribble, sprinter, clip_model, clip_processor, device)
        print(f'Target CLIP: {target_clip.shape}')
        if len(group_imgs) > 8:   # age: one gallery per age -- keep five along the range
            keys = list(group_imgs)
            group_imgs = {k: group_imgs[k] for k in keys[::max(1, len(keys) // 4)]}
        save_target_previews(group_imgs, out_dir, n_preview=args.n_preview)

        def candidate(i):
            cand = run_sdedit(source_scribble, seed=SEED + i, architect=architect, device=device,
                              cfg_scale=0.0, n_steps_total=cfg['n_steps'],
                              sdedit_start=cfg['sdedit_start'], prompt='')
            imgs = gen_images(sprinter, cand, cfg['neutral_prompt'],
                              cfg['n_eval_search'], cfg['controlnet_scale'], seed=SEED)
            return cand, mmd_report(compute_mmd(clip_embed(imgs, clip_model, clip_processor, device),
                                                target_clip)).item()

        print('\nMeasuring the first SDEdit candidate...')
        t0 = time.time()
        first_cand, first_mmd = candidate(0)
        sec_per_candidate = time.time() - t0
        n_candidates = max(1, int(lgd_minutes * 60 / sec_per_candidate))   # time-matched to the MLGD-F run
        if args.max_candidates:
            n_candidates = min(n_candidates, args.max_candidates)
        print(f'  {sec_per_candidate:.1f}s per candidate -> budget {lgd_minutes:.1f} min -> {n_candidates} candidates')
        wandb.log({'sec_per_candidate': sec_per_candidate, 'n_candidates': n_candidates})

        candidate_mmds, best = [first_mmd], first_cand
        tqdm.write(f'  [1/{n_candidates}] seed={SEED}  MMD={first_mmd:.5f}')
        wandb.log({'candidate_mmd': first_mmd, 'candidate_idx': 1})
        for i in tqdm(range(1, n_candidates), desc='SDEdit candidates'):
            cand, mmd = candidate(i)
            if mmd < min(candidate_mmds):
                best = cand
            candidate_mmds.append(mmd)
            tqdm.write(f'  [{i+1}/{n_candidates}] seed={SEED+i}  MMD={mmd:.5f}')
            wandb.log({'candidate_mmd': mmd, 'candidate_idx': i + 1})
        best_idx = int(np.argmin(candidate_mmds))
        scribbles['sdedit_best'] = best
        print(f'\nBest: seed={SEED + best_idx}  MMD={candidate_mmds[best_idx]:.5f}')
        save_sdedit_search_plot(candidate_mmds, best_idx, out_dir)
        meta.update(lgd_cm_minutes=lgd_minutes, n_candidates=n_candidates,
                    sec_per_candidate=round(sec_per_candidate, 2), best_candidate_idx=best_idx,
                    best_candidate_seed=SEED + best_idx, best_candidate_mmd=candidate_mmds[best_idx],
                    n_eval_search=cfg['n_eval_search'], all_candidate_mmds=candidate_mmds)
        wandb.log({'best_candidate_idx': best_idx, 'best_candidate_mmd': candidate_mmds[best_idx],
                   'search_curve': wandb.plot.line_series(
                       xs=list(range(len(candidate_mmds))), ys=[candidate_mmds], keys=['MMD'],
                       title='SDEdit candidate search', xname='candidate')})

    # ── Save the scribbles, and a preview of every scribble ──
    print('\nSaving outputs...')
    save_scribble_grid({'source': source_scribble, **scribbles,
                        **({'mlgd_f': mlgd_scribble} if mlgd_scribble else {})}, out_dir)
    preview_log = {}
    for name, scr in {'source': source_scribble, **scribbles,
                      **({'mlgd_f': mlgd_scribble} if mlgd_scribble else {})}.items():
        imgs = gen_images(sprinter, scr, cfg['neutral_prompt'], args.n_preview,
                          cfg['controlnet_scale'], seed=SEED)
        plot_row(imgs, f'{scenario}: {name}', count=args.n_preview,
                 save_path=str(out_dir / f'preview_{name}.png'))
        preview_log[f'scribble/{name}'] = wandb.Image(scr)
        preview_log[f'preview/{name}'] = wandb.Image(str(out_dir / f'preview_{name}.png'))
        print(f'  Saved preview_{name}.png')
    wandb.log(preview_log)

    meta['wandb'] = {'project': run.project, 'name': run.name, 'id': run.id, 'url': run.url}
    with open(out_dir / 'baselines_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('  Saved baselines_meta.json')

    wandb.finish()
    print(f'\n✅ Done: {", ".join(scribbles)} in {out_dir}')


if __name__ == '__main__':
    main()
