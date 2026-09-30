# Provenance — four_targets

## Code

Source: `orineo1/conditional-matching-paper`:

| | |
|---|---|
| branch | `claude/sd-shared-controlnet-hvt-uniform` |
| commit | `cb8bc915b3d7f2f6107c1f14457e3d056188704f` |
| date | 2026-09-24 |
| path | `SD_cond_SD_controlnet/` |

**Why this branch.** It is the newest SD branch and the only one carrying the complete
parameterised pipeline the paper's protocol calls for: uniform back-selection
(`--backsel_rule uniform`, `--backsel_k`) and a single `--controlnet_scale` threaded
through target construction, the guided step and evaluation alike. The release-
shaped branch `CLEAN_SD_V2` (`cab3f11`, 2026-05-06) has the same module layout
(`scripts/run_mlgd_f.py`, `src/generation.py`, `src/metrics.py`) but no selection rule and
a hard-coded conditioning scale in the guided step, so it cannot express the configuration
in Table 14.

The code lives in `cdm/models/sd/` at the repository root. See `cdm/models/sd/README.md` for what is
carried there versus what is called from the shared packages.

## Reference numbers in `results/`

`table3_sd_results.md` and `table15_sd_swd.md` are transcribed from the paper
(Table 3 `tab:sd_results`; Table 15 `tab:sd_swd`, Appendix D.6). They are the **declared
reference** for this experiment.

## Outputs

`mlgdf_runs.md`, `eval_table.md` and the `full_<Scenario>_seed<S>_<jobid>/` folders, when
present, are runs of this code copied from `outputs/` by
`experiments/stable_diffusion/export_results.py`; each carries its commit, command and W&B
link in `run_info.txt` / `metrics.json`, and its `baselines/` and `eval/` (the N = 2,000
evaluation, the protocol of Tables 3 and 15).

`outputs/` is gitignored and holds only what a local run writes; nothing in it is a
published number.
