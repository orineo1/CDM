> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# No published Gaussian number

App. B.3 states that `SimulationParameters/ldm_10D_full.txt` "is the one used for the
selection-rule comparison of Appendix G". Appendix G itself contains definitions plus
Table 17 `tab:select-runtime`, which is an SDXL micro-benchmark and belongs to the SDXL
area. There is therefore no Gaussian float for this directory to declare.

`scripts/control_subsample.py` is the control that runs on this setting. It needs the
`conditional-matching-paper` simulations directory (`CMP_SIM_DIR`) and downloads two
checkpoints from HuggingFace on first use.
