"""The synthetic-experiment numbers as printed in the paper, for compare_to_paper.py.

Transcribed from the paper source; each block names its table or figure. Values are
(mean, std) where the paper prints "mean +- std", otherwise a float or a string. The
"refined" column of Table 8 is Table 1 reprinted and is not repeated here.
"""

SCENARIOS = ["1x1", "1x5", "1x10", "5x1", "5x5", "5x10", "10x1", "10x5", "10x10"]
METHODS = ["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"]

# Table 1 (tab:sim_main): TV(P(Y | X = x_hat), G), mean +- std over five runs.
TABLE1 = {
    "1x1":   {"MLGD": (0.02, 0.01), "MLGD-F": (0.08, 0.01), "PMLGD": (0.02, 0.01), "PMLGD-F": (0.08, 0.02)},
    "1x5":   {"MLGD": (0.02, 0.01), "MLGD-F": (0.08, 0.02), "PMLGD": (0.01, 0.01), "PMLGD-F": (0.07, 0.03)},
    "1x10":  {"MLGD": (0.07, 0.07), "MLGD-F": (0.38, 0.05), "PMLGD": (0.06, 0.05), "PMLGD-F": (0.37, 0.07)},
    "5x1":   {"MLGD": (0.08, 0.01), "MLGD-F": (0.25, 0.09), "PMLGD": (0.17, 0.11), "PMLGD-F": (0.27, 0.09)},
    "5x5":   {"MLGD": (0.06, 0.02), "MLGD-F": (0.50, 0.05), "PMLGD": (0.17, 0.19), "PMLGD-F": (0.48, 0.03)},
    "5x10":  {"MLGD": (0.14, 0.03), "MLGD-F": (0.41, 0.04), "PMLGD": (0.16, 0.06), "PMLGD-F": (0.45, 0.03)},
    "10x1":  {"MLGD": (0.49, 0.08), "MLGD-F": (0.56, 0.03), "PMLGD": (0.99, 0.02), "PMLGD-F": (0.78, 0.22)},
    "10x5":  {"MLGD": (0.13, 0.04), "MLGD-F": (0.30, 0.01), "PMLGD": (0.18, 0.12), "PMLGD-F": (0.30, 0.03)},
    "10x10": {"MLGD": (0.20, 0.05), "MLGD-F": (0.92, 0.02), "PMLGD": (0.43, 0.13), "PMLGD-F": (0.81, 0.12)},
}

# Table 5 (tab:model_quality): score-net relative Fisher, prior SWD, conditional DDPM at x* and
# at random x, iCT at x* and at random x.
TABLE5_COLUMNS = ["fisher", "prior_swd", "ddpm_xstar", "ddpm_random", "ict_xstar", "ict_random"]
TABLE5 = {
    "1x1":   (0.001, 0.020, 0.012, 0.166, 0.041, 1.234),
    "1x5":   (0.005, 0.036, 0.041, 0.043, 0.166, 0.161),
    "1x10":  (0.003, 0.036, 0.032, 0.041, 0.105, 0.097),
    "5x1":   (0.006, 0.026, 0.082, 0.193, 0.112, 0.882),
    "5x5":   (0.007, 0.030, 0.048, 0.097, 0.213, 0.258),
    "5x10":  (0.010, 0.033, 0.040, 0.088, 0.093, 0.164),
    "10x1":  (0.011, 0.036, 0.990, 0.766, 0.563, 1.673),
    "10x5":  (0.014, 0.029, 0.038, 0.181, 0.165, 0.406),
    "10x10": (0.017, 0.032, 0.057, 0.142, 0.279, 0.401),
}

# tab:sim_full (the appendix companion of Table 1; formerly Table 8 tab:sim_ablation): TV without
# refinement, mean +- std over five runs; arms "norefine", "annealed", "oracle". Only the first
# five scenarios were transcribed, from the earlier version of the paper; the ||x_hat - x*|| and
# relative L2 columns of the current table are not transcribed.
TABLE8 = {
    "1x1": {"MLGD":    {"norefine": (0.01, 0.01), "oracle": (0.02, 0.01)},
            "MLGD-F":  {"norefine": (0.08, 0.05)},
            "PMLGD":   {"norefine": (0.02, 0.02), "annealed": (0.02, 0.01), "oracle": (0.02, 0.01)},
            "PMLGD-F": {"norefine": (0.07, 0.01), "annealed": (0.10, 0.03)}},
    "1x5": {"MLGD":    {"norefine": (0.02, 0.01), "oracle": (0.02, 0.01)},
            "MLGD-F":  {"norefine": (0.08, 0.05)},
            "PMLGD":   {"norefine": (0.03, 0.03), "annealed": (0.03, 0.03), "oracle": (0.02, 0.01)},
            "PMLGD-F": {"norefine": (0.07, 0.04), "annealed": (0.07, 0.04)}},
    "1x10": {"MLGD":    {"norefine": (0.12, 0.08), "oracle": (0.07, 0.04)},
             "MLGD-F":  {"norefine": (0.42, 0.08)},
             "PMLGD":   {"norefine": (0.12, 0.13), "annealed": (0.12, 0.13), "oracle": (0.06, 0.05)},
             "PMLGD-F": {"norefine": (0.41, 0.06), "annealed": (0.41, 0.06)}},
    "5x1": {"MLGD":    {"norefine": (0.10, 0.06), "oracle": (0.17, 0.17)},
            "MLGD-F":  {"norefine": (0.22, 0.11)},
            "PMLGD":   {"norefine": (0.19, 0.09), "annealed": (0.12, 0.03), "oracle": (0.51, 0.02)},
            "PMLGD-F": {"norefine": (0.30, 0.10), "annealed": (0.27, 0.06)}},
    "5x5": {"MLGD":    {"norefine": (0.28, 0.09), "oracle": (0.24, 0.15)},
            "MLGD-F":  {"norefine": (0.49, 0.03)},
            "PMLGD":   {"norefine": (0.43, 0.28), "annealed": (0.23, 0.07), "oracle": (0.63, 0.25)}},
}

# Table 9 (tab:nonstar_results): R = 25 restarts per arm.
TABLE9 = {                     # arm: (% at x_bi, time mean, time std)
    "MMD_n250": (96.0, 7.40, 1.06),
    "L2_n250":  (0.0, 4.31, 0.60),
    "L2_n1":    (12.0, 1.65, 0.46),
}

# Training times (tab:training_times, commented out in the current paper): approximate minutes on an L4.
TABLE10 = {                    # (params file, network): minutes
    ("mog_2d_full", "uncond"): 11, ("mog_2d_full", "diffusion"): 12, ("mog_2d_full", "cm"): 3,
    ("mog_10d_full", "uncond"): 240, ("mog_10d_full", "diffusion"): 240, ("mog_10d_full", "cm"): 9,
}

# Figure 3 (fig:smc_correction): TV of the histogram to Q_beta, plain and corrected.
FIGURE3 = {"2.0": (0.17, 0.03), "8.0": (0.19, 0.03), "16.0": (0.56, 0.06)}

# Figure 1 (fig:sim_2d), panels (a)/(c): restarts (out of ten) that reach the x* basin.
FIGURE1_HITS = {"MLGD": 1, "MLGD-F": 0, "PMLGD": 6, "PMLGD-F": 7}
