# `jacobian_error/`

The Jacobian-fidelity diagnostic of the theory appendix: how closely the few-step
student (SDXL-Lightning) matches its multi-step teacher (SDXL-Base) in output and in
Jacobian with respect to the conditioning, per prompt (the `r_s`, `r_g` ratios and the
outlier analysis).

| file | what it is |
|---|---|
| `eps_g_experiment.ipynb` | the full experiment: loads both models from Hugging Face, computes the output and Jacobian gaps per prompt, and draws the figures |

The notebook is self-contained (only `torch`, `diffusers`, `huggingface_hub`,
`safetensors`, `numpy`, `pandas`, `matplotlib`); it does not import the `cdm` package.

## Compute

The paper's run used one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). It evaluated
N = 100 prompts with D = 10 VJP directions each, two reverse-mode passes per direction
(one through the teacher, one through the student), about 3 hours of wall-clock time.
