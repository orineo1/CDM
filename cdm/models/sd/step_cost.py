# Port of orineo1/conditional-matching-paper SD_cond_SD_controlnet/notebooks/measure_dps_step_memory.ipynb
# (memory) and of scripts/time_dps_step.py on branch test-witness-result (the per-step timing of the
# selection table), measured the same way: random latents and target embeddings, one warm-up call
# of the same configuration, then the timed repeats. Run as:
#     bash experiments/stable_diffusion/step_cost/run.sh
#     python -m cdm.models.sd.step_cost --help

"""
step_cost.py -- GPU memory and time of one MLGD-F guidance step (paper: ablation, Turbo vs
Base; selection rules, per-step time).

One guidance step as in run_mlgd_f.py (CFG noise prediction at a mid-schedule timestep,
pred_x0, checkpointed VAE decode, run_dps_step_clip), on random latents and --n_target random
target CLIP vectors: the content does not change the cost. Measured for every --num_variations
(n_cond) and each selection rule, All (every sample backpropagated, n_sel = n_cond) and
Uniform (n_sel = --uniform_fraction * n_cond), with one untimed warm-up call of the same
configuration and then --n_repeats timed calls.

Per configuration, peak memory decomposed as M_models + M_fixed + M_var
(M_fixed: UNet forward, pred_x0, VAE decode; M_var: the n_cond Sprinter variations of K = 2
steps, the MMD and the backward), and the step time split the same way (t_fixed, t_var).

The SDXL-Base projection (K* = --base_steps instead of K = --turbo_steps): the two models
share the architecture, so models and the fixed part are unchanged and the per-sample part
scales with K*/K: M_var x K*/K and t_var x K*/K. A full run is --n_steps - --start_step
guided steps. This is an extrapolation, not a measurement.

Fast (non-deterministic) GPU kernels by default: a cost measurement needs no bit-identical
results, and the deterministic ones (--deterministic; what run_mlgd_f uses) are slower and
use more memory.

Writes to --output_dir: step_cost.json (every measurement, the settings, GPU, model revisions,
versions) and step_cost.md (the two tables: memory and run time, Turbo vs Base; per-step
time, All vs Uniform).
"""

import argparse
import gc
import json
import os
import time

import numpy as np

VARIATION_PROMPT = "a superrealistic professional photograph of"
GB = 1e9


# ---------------------------------------------------------------------------
# Bookkeeping (no GPU)
# ---------------------------------------------------------------------------

def summarize(reps, n_cond, n_sel, rule, gpu_total, guided_steps, ratio):
    """One configuration's repeats -> means, std of the step time, and the Base projection."""
    step = np.array([r["t_fixed"] + r["t_var"] for r in reps])
    t_fixed = float(np.mean([r["t_fixed"] for r in reps]))
    t_var = float(np.mean([r["t_var"] for r in reps]))
    m = {k: max(r[k] for r in reps) for k in ("models", "fixed", "var", "peak")}
    base_peak = m["models"] + m["fixed"] + m["var"] * ratio
    base_step = t_fixed + t_var * ratio
    return {"rule": rule, "n_cond": n_cond, "n_sel": n_sel, "n_repeats": len(reps),
            "step_s_mean": float(step.mean()), "step_s_std": float(step.std()),
            "t_fixed_s": t_fixed, "t_var_s": t_var,
            "models_gb": m["models"] / GB, "fixed_gb": m["fixed"] / GB, "var_gb": m["var"] / GB,
            "peak_gb": m["peak"] / GB, "gpu_gb": gpu_total / GB,
            "run_h": float(step.mean()) * guided_steps / 3600,
            "base": {"peak_gb": base_peak / GB, "x_gpu": base_peak / gpu_total,
                     "step_s": base_step, "run_h": base_step * guided_steps / 3600},
            "repeats": reps}


def memory_table(rows, gpu, guided_steps, turbo_steps, base_steps):
    lines = [f"## Memory and run time, All selection ({gpu})", "",
             f"Turbo (K = {turbo_steps}) measured; Base (K* = {base_steps}) projected: per-sample part "
             f"x {base_steps}/{turbo_steps}, models and fixed part unchanged. Run = {guided_steps} guided steps.", "",
             "| n_cond | models (GB) | fixed (GB) | samples (GB) | Turbo peak (GB) | Base peak (GB, proj.) "
             "| Base / GPU memory | Turbo run (h) | Base run (h, proj.) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["rule"] != "all":
            continue
        if r.get("oom"):
            lines.append(f"| {r['n_cond']} | | | | OOM | | | | |")
            continue
        b = r["base"]
        lines.append(f"| {r['n_cond']} | {r['models_gb']:.1f} | {r['fixed_gb']:.1f} | {r['var_gb']:.1f} | "
                     f"{r['peak_gb']:.1f} | {b['peak_gb']:.0f} | {b['x_gpu']:.1f}x | {r['run_h']:.1f} | {b['run_h']:.0f} |")
    return "\n".join(lines)


def runtime_table(rows, gpu, fraction):
    by = {(r["n_cond"], r["rule"]): r for r in rows}
    fmt = lambda r: "OOM" if r is None or r.get("oom") else f"{r['step_s_mean']:.2f} ± {r['step_s_std']:.2f}"
    reps = next((r["n_repeats"] for r in rows if not r.get("oom")), 0)
    lines = [f"## Per-step time, All vs Uniform ({gpu})", "",
             f"One guidance step in isolation, mean ± std over {reps} repeats.", "",
             f"| n_cond | All (s/step) | Uniform, n_sel/n_cond = {fraction:g} (s/step) | saving |",
             "|---|---|---|---|"]
    for n in sorted({r["n_cond"] for r in rows}):
        a, u = by.get((n, "all")), by.get((n, "uniform"))
        ok = a and u and not a.get("oom") and not u.get("oom")
        saving = f"{100 * (1 - u['step_s_mean'] / a['step_s_mean']):.0f}%" if ok else ""
        lines.append(f"| {n} | {fmt(a)} | {fmt(u)} | {saving} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU part
# ---------------------------------------------------------------------------

def _reset():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_once(ctx, n_cond, n_sel, args):
    """One guidance step; bytes and seconds."""
    import torch
    import torch.utils.checkpoint
    from cdm.models.sd.generation import compute_pred_x0_direct, predict_noise_cfg, run_dps_step_clip
    from cdm.models.sd.metrics import compute_mmd

    a, s = ctx["architect"], ctx["sprinter"]
    _reset()
    models = torch.cuda.memory_allocated()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    latents = torch.randn(ctx["latents_shape"], device="cuda", dtype=torch.float16)
    latents_step = latents.detach().requires_grad_(True)
    noise_pred = predict_noise_cfg(a.unet, a.scheduler, latents_step, ctx["t"], ctx["cfg_states"],
                                   ctx["added_cond"], args.guidance_scale)
    pred_x0 = compute_pred_x0_direct(a.scheduler, noise_pred, ctx["t"], latents_step)
    px = torch.utils.checkpoint.checkpoint(lambda lat: a.vae.decode(lat.to(a.vae.dtype)).sample,
                                           pred_x0 / a.vae.config.scaling_factor, use_reentrant=False)
    px = torch.clamp((px + 1.0) / 2.0, 0.0, 1.0)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    fixed_peak = torch.cuda.max_memory_allocated()
    grad, *_ = run_dps_step_clip(
        latents=latents, latents_step=latents_step, noise_pred=noise_pred, pixel_x0_norm=px,
        sprinter=s, all_clip_embeddings=ctx["targets"], num_variations=n_cond, variation_batch_size=1,
        base_zeta_prime=args.base_zeta, clip_model=ctx["clip_model"], clip_processor=ctx["clip_processor"],
        vae=s.vae, vae_scaling_factor=s.vae.config.scaling_factor, variation_prompt=args.variation_prompt,
        loss_fn=compute_mmd, controlnet_scale=args.controlnet_scale,
        backsel_k=None if n_sel == n_cond else n_sel, backsel_rule="uniform")
    torch.cuda.synchronize(); t2 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    del grad, latents, latents_step, noise_pred, pred_x0, px
    return {"t_fixed": t1 - t0, "t_var": t2 - t1, "models": models, "fixed": fixed_peak - models,
            "var": peak - fixed_peak, "peak": peak}


def setup(args, device):
    """Models, the CFG text state, a mid-schedule timestep, random target CLIP vectors."""
    import torch
    from cdm.models.sd.clip_utils import load_clip_model
    from cdm.models.sd.models import load_models, setup_gradient_checkpointing

    architect, sprinter = load_models(device, architect_revision=args.architect_revision,
                                      sprinter_revision=args.sprinter_revision,
                                      controlnet_revision=args.controlnet_revision)
    clip_model, clip_processor = load_clip_model(device, revision=args.clip_revision)
    setup_gradient_checkpointing(architect, sprinter)
    sprinter.vae.to(dtype=torch.float32)
    with torch.no_grad():
        pe, npe, ppe, nppe = architect.encode_prompt(prompt=args.variation_prompt, negative_prompt="",
                                                     device=device, do_classifier_free_guidance=True,
                                                     num_images_per_prompt=1)
    architect.scheduler.set_timesteps(args.n_steps, device=device)
    return {"architect": architect, "sprinter": sprinter, "clip_model": clip_model,
            "clip_processor": clip_processor, "t": architect.scheduler.timesteps[args.start_step],
            "targets": torch.randn(args.n_target, 768, device=device),
            "latents_shape": (1, architect.unet.config.in_channels, 64, 64),
            "cfg_states": torch.cat([npe, pe], dim=0),
            "added_cond": {"text_embeds": torch.cat([nppe, ppe], dim=0),
                           "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype,
                                                    device=device).repeat(2, 1)}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_variations", type=int, nargs="+", default=[100, 120], help="n_cond values")
    p.add_argument("--rules", nargs="+", default=["all", "uniform"], choices=["all", "uniform"])
    p.add_argument("--uniform_fraction", type=float, default=0.5, help="n_sel / n_cond of the Uniform rule")
    p.add_argument("--n_warmup", type=int, default=1, help="untimed calls of each configuration first")
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--n_target", type=int, default=250,
                   help="random target CLIP vectors (only the MMD matrix size depends on it)")
    p.add_argument("--n_steps", type=int, default=250)
    p.add_argument("--start_step", type=int, default=125, help="guided steps per run: n_steps - start_step")
    p.add_argument("--turbo_steps", type=int, default=2)
    p.add_argument("--base_steps", type=int, default=30)
    p.add_argument("--base_zeta", type=float, default=7.5)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--variation_prompt", type=str, default=VARIATION_PROMPT)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic kernels, as run_mlgd_f uses (slower, more memory); default: fast")
    for m in ("architect", "sprinter", "controlnet", "hed", "clip"):
        p.add_argument(f"--{m}_revision", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="sdxl-step-cost")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from cdm.models.sd.clip_utils import CLIP_MODEL_ID
    from cdm.models.sd.models import (DEFAULT_ARCHITECT_MODEL_ID, DEFAULT_CONTROLNET_MODEL_ID,
                                      DEFAULT_SPRINTER_MODEL_ID, make_deterministic,
                                      resolved_revision)
    from cdm.models.sd.run_mlgd_f import library_versions

    if not torch.cuda.is_available():
        raise SystemExit("step_cost measures GPU memory: needs a CUDA GPU")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.deterministic:
        make_deterministic(args.seed)
    else:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    gpu = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    guided = args.n_steps - args.start_step
    ratio = args.base_steps / args.turbo_steps
    print(f"{gpu}, {gpu_total / GB:.0f} GB; deterministic kernels: {args.deterministic}", flush=True)

    ctx = setup(args, "cuda")
    revisions = {DEFAULT_ARCHITECT_MODEL_ID: resolved_revision(DEFAULT_ARCHITECT_MODEL_ID, args.architect_revision),
                 DEFAULT_SPRINTER_MODEL_ID: resolved_revision(DEFAULT_SPRINTER_MODEL_ID, args.sprinter_revision),
                 DEFAULT_CONTROLNET_MODEL_ID: resolved_revision(DEFAULT_CONTROLNET_MODEL_ID, args.controlnet_revision),
                 CLIP_MODEL_ID: resolved_revision(CLIP_MODEL_ID, args.clip_revision)}
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                         config={**vars(args), "gpu": gpu, "model_revisions": revisions,
                                 "versions": library_versions()})

    rows = []
    for n in args.num_variations:
        for rule in args.rules:
            n_sel = n if rule == "all" else max(1, round(args.uniform_fraction * n))
            print(f"n_cond={n}, {rule} (n_sel={n_sel}), {args.n_repeats} repeats...", flush=True)
            try:
                for _ in range(args.n_warmup):
                    measure_once(ctx, n, n_sel, args)
                reps = []
                for k in range(args.n_repeats):
                    reps.append(measure_once(ctx, n, n_sel, args))
                    print(f"  {k + 1}: {reps[-1]['t_fixed'] + reps[-1]['t_var']:.2f} s, "
                          f"peak {reps[-1]['peak'] / GB:.1f} GB", flush=True)
                rows.append(summarize(reps, n, n_sel, rule, gpu_total, guided, ratio))
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True)
                _reset()
                rows.append({"rule": rule, "n_cond": n, "n_sel": n_sel, "oom": True})

    md = (f"# One MLGD-F guidance step: {gpu}, deterministic kernels: {args.deterministic}\n\n"
          + memory_table(rows, gpu, guided, args.turbo_steps, args.base_steps) + "\n\n"
          + runtime_table(rows, gpu, args.uniform_fraction) + "\n")
    with open(os.path.join(args.output_dir, "step_cost.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.output_dir, "step_cost.json"), "w") as f:
        json.dump({"gpu": gpu, "gpu_gb": gpu_total / GB, "deterministic": args.deterministic,
                   "guided_steps": guided, "base_over_turbo": ratio, "rows": rows,
                   "settings": {k: v for k, v in vars(args).items() if not k.endswith("_revision")},
                   "model_revisions": revisions, "versions": library_versions(),
                   "wandb": ({"project": run.project, "name": run.name, "url": run.url} if run else None)},
                  f, indent=2)
    print("\n" + md, flush=True)
    if run:
        import wandb
        wandb.log({"step_cost": wandb.Table(
            columns=["rule", "n_cond", "n_sel", "step_s_mean", "step_s_std", "peak_gb", "base_peak_gb", "base_run_h"],
            data=[[r["rule"], r["n_cond"], r["n_sel"], r.get("step_s_mean"), r.get("step_s_std"),
                   r.get("peak_gb"), (r.get("base") or {}).get("peak_gb"), (r.get("base") or {}).get("run_h")]
                  for r in rows])})
        wandb.finish()
    print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
