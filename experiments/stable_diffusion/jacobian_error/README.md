# `jacobian_error/`

The Jacobian-fidelity diagnostic of the theory appendix: how closely the few-step
student (SDXL-Lightning) matches its multi-step teacher (SDXL-Base) in output and in
Jacobian with respect to the conditioning, per prompt (the `r_s`, `r_g` ratios and the
outlier analysis).

| file | what it is |
|---|---|
| `eps_g_experiment.ipynb` | the full experiment: loads both models from Hugging Face, computes the output and Jacobian gaps per prompt, and draws the figures |

**Two definitions of $\hat r_g$** (both computed in §6, compared in §7 and in `results/rg_l2_vs_old.md`):
*old* = per prompt the mean over the D directions of $\|v_d^\top\Delta J_i\|$ (and of $\|v_d^\top\partial_x f_\phi\|$), then the RMS over prompts;
*L²* = the RMS over all (prompt, direction) pairs of the same norms, which estimates $\|\Delta J\|_F/\|J\|_F$ with no Jensen factor.
**The paper reports the L² version** (figure `rms_combined_l2.pdf`). Both are computed from
`results/vjp_results_lightning_l2.csv` (per-direction columns `vjp_kk` and `jphi_kk`); the old figure is still written
to `rms_combined.pdf`, and the earlier run's `results/vjp_results_lightning.csv` is not touched. `r_s` is the same in both.
Set `SMOKE = True` at the top of §1 for a quick end-to-end check of the code (2 prompts, 2 directions, `_smoke` file suffix).

The notebook is self-contained (only `torch`, `diffusers`, `huggingface_hub`,
`safetensors`, `numpy`, `pandas`, `matplotlib`); it does not import the `cdm` package.

## Compute

The paper's run used one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). It evaluated
N = 100 prompts with D = 10 VJP directions each, two reverse-mode passes per direction
(one through the teacher, one through the student), about 3 hours of wall-clock time.
