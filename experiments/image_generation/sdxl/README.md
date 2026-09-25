# `experiments/image_generation/sdxl/`

The Stable Diffusion experiment. One directory per experiment, each with `README.md`
(what it produces, the command, runtime, GPU), `run.sh` (explicit tiers, every flag
literal), `config/`, `results/` (reference numbers **imported** with provenance) and
`outputs/` (local runs, gitignored). The two are never mixed.

| directory | paper artifact | entry point | ran here |
|---|---|---|---|
| `mlgd_f_four_targets` | Table 3, Table 15 | `sdxl.run_mlgd_f`, `sdxl.eval_baselines` | `check` tier only |

**The code lives in `sdxl/` at the repository root**, ported from
`orineo1/conditional-matching-paper` @ `cb8bc91` so this repository is self-contained —
nothing points at another repo at runtime. `sdxl/README.md` says what was copied in and
what is called from the shared packages; in short, the SDXL + ControlNet-Scribble
pipeline, HED extraction, SDEdit init, target construction, CLIP evaluation and the
baselines are SD-specific and carried here, while the **MMD estimator is the shared
`losses.functions.compute_mmd`** and is called, not duplicated.

SDXL does not use `models/DiffusionModel`; it uses Stable Diffusion's own pipelines.

**Nothing here was re-run.** Every number in a `results/` directory is imported from the
paper. This experiment requires a GPU (L40S 48 GB) and large model downloads, and the
machine this was built on has neither a GPU nor network access, so only the CPU `check`
tier was executed. Each README says which tier was actually run.

## Not covered here

These inventory rows belong to the SD experiment but have **no ported entry point**. The
upstream code for them is in notebooks under `SD_cond_SD_controlnet/notebooks/` at the
same commit, which were not ported:

| row | artifact | upstream notebook |
|---|---|---|
| SD-09 | Fig. 10, saliency + correlation | `gender_saliency_eval.ipynb` |
| SD-13, SD-14 | Table 17 step timing; memory ablation | `measure_dps_step_memory.ipynb` |
| SD-16, SD-17 | Table 18, Fig. 20 Jacobian fidelity | `eps_g_experiment.ipynb` |
| SD-20 | the 2,000-image evaluation protocol | `eval_all_experiments.ipynb` |

`sdxl/analysis.py` and `sdxl/visualization.py` carry the scribble-diff heatmap and the
step plots (SD-08), but the saliency computation itself is not among them.
