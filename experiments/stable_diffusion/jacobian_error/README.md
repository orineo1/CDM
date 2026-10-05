# jacobian_error

The Jacobian-fidelity diagnostic: how closely the few-step student (SDXL-Lightning, 4 steps) matches its teacher
(SDXL Base, 20 DDIM steps) in output and in its Jacobian with respect to the conditioning, per prompt (the ratios
`r_s`, `r_g` and the outlier analysis).

`eps_g_experiment.ipynb` loads both models from Hugging Face, computes the gaps for N = 100 prompts with D = 10 VJP
directions each, and draws the figures. It is self-contained (`torch`, `diffusers`, `huggingface_hub`,
`safetensors`, `numpy`, `pandas`, `matplotlib`) and does not import `cdm`.

The paper's run used one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB) for about 3 hours.
