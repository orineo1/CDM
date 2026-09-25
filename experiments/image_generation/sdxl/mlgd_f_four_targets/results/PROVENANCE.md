# Provenance — mlgd_f_four_targets

## Code

Ported from `orineo1/conditional-matching-paper`:

| | |
|---|---|
| branch | `claude/sd-shared-controlnet-hvt-uniform` |
| commit | `cb8bc915b3d7f2f6107c1f14457e3d056188704f` |
| date | 2026-09-24 |
| path | `SD_cond_SD_controlnet/` |

**Why this branch.** It is the newest SD branch and the only one carrying the complete
parameterised pipeline the paper's protocol calls for: uniform back-selection
(`--backsel_rule uniform`, `--backsel_k`) and a single `--controlnet_scale` threaded
through target construction, the guided step and evaluation alike. The earlier release-
shaped branch `CLEAN_SD_V2` (`cab3f11`, 2026-05-06) has the same module layout
(`scripts/run_mlgd_f.py`, `src/generation.py`, `src/metrics.py`) but no selection rule and
a hard-coded conditioning scale in the guided step, so it cannot express the configuration
in Table 14.

Ported into `sdxl/` at the repository root. See `sdxl/README.md` for what was copied
versus what is called from the shared packages.

## Reference numbers in `results/`

`table3_sd_results.md` and `table15_sd_swd.md` are transcribed from the paper
(Table 3 `tab:sd_results`; Table 15 `tab:sd_swd`, Appendix D.6). They are the **declared
reference** for this experiment.

They were **imported, not reproduced**. Nothing in this repository re-ran them, and no
number in `results/` was computed here.

## What was produced locally

Nothing. `outputs/` is empty. See `../README.md` for what could and could not be
executed on the machine this was built on.
