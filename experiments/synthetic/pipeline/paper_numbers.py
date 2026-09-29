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

# tab:sim_full (the appendix companion of Table 1, from aistats2027/tables/sim_appendix_rows.tex):
# per scenario and method, "dist" = ||x_hat - x*|| in sd units and "l2rel" = relative L2, both after
# refinement, and the TV without refinement of the "norefine", "annealed" and "oracle" arms.
SIM_FULL = {
    "1x1": {
        "MLGD": {'dist': (0.004, 0.003), 'l2rel': (0.003, 0.003), 'norefine': (0.01, 0.01), 'oracle': (0.02, 0.01)},
        "MLGD-F": {'dist': (0.019, 0.003), 'l2rel': (0.039, 0.011), 'norefine': (0.08, 0.05)},
        "PMLGD": {'dist': (0.006, 0.003), 'l2rel': (0.005, 0.004), 'norefine': (0.02, 0.02), 'annealed': (0.02, 0.01), 'oracle': (0.02, 0.01)},
        "PMLGD-F": {'dist': (0.02, 0.006), 'l2rel': (0.046, 0.022), 'norefine': (0.07, 0.01), 'annealed': (0.1, 0.03)},
    },
    "1x5": {
        "MLGD": {'dist': (0.018, 0.011), 'l2rel': (0.002, 0.002), 'norefine': (0.02, 0.01), 'oracle': (0.02, 0.01)},
        "MLGD-F": {'dist': (0.078, 0.022), 'l2rel': (0.04, 0.02), 'norefine': (0.08, 0.05)},
        "PMLGD": {'dist': (0.013, 0.01), 'l2rel': (0.001, 0.002), 'norefine': (0.03, 0.03), 'annealed': (0.03, 0.03), 'oracle': (0.02, 0.01)},
        "PMLGD-F": {'dist': (0.068, 0.029), 'l2rel': (0.034, 0.025), 'norefine': (0.07, 0.04), 'annealed': (0.07, 0.04)},
    },
    "1x10": {
        "MLGD": {'dist': (0.015, 0.014), 'l2rel': (0.054, 0.087), 'norefine': (0.12, 0.08), 'oracle': (0.07, 0.04)},
        "MLGD-F": {'dist': (0.081, 0.011), 'l2rel': (0.684, 0.138), 'norefine': (0.42, 0.08)},
        "PMLGD": {'dist': (0.012, 0.009), 'l2rel': (0.03, 0.036), 'norefine': (0.12, 0.13), 'annealed': (0.12, 0.13), 'oracle': (0.06, 0.05)},
        "PMLGD-F": {'dist': (0.078, 0.017), 'l2rel': (0.644, 0.204), 'norefine': (0.41, 0.06), 'annealed': (0.41, 0.06)},
    },
    "5x1": {
        "MLGD": {'dist': (0.938, 0.249), 'l2rel': (0.028, 0.007), 'norefine': (0.1, 0.06), 'oracle': (0.17, 0.17)},
        "MLGD-F": {'dist': (1.336, 0.636), 'l2rel': (0.297, 0.222), 'norefine': (0.22, 0.11)},
        "PMLGD": {'dist': (1.023, 0.483), 'l2rel': (0.184, 0.228), 'norefine': (0.19, 0.09), 'annealed': (0.12, 0.03), 'oracle': (0.51, 0.02)},
        "PMLGD-F": {'dist': (1.427, 0.691), 'l2rel': (0.357, 0.227), 'norefine': (0.3, 0.1), 'annealed': (0.27, 0.06)},
    },
    "5x5": {
        "MLGD": {'dist': (0.038, 0.023), 'l2rel': (0.014, 0.007), 'norefine': (0.28, 0.09), 'oracle': (0.24, 0.15)},
        "MLGD-F": {'dist': (0.479, 0.016), 'l2rel': (0.465, 0.153), 'norefine': (0.49, 0.03)},
        "PMLGD": {'dist': (0.194, 0.297), 'l2rel': (0.121, 0.196), 'norefine': (0.43, 0.28), 'annealed': (0.23, 0.07), 'oracle': (0.63, 0.25)},
        "PMLGD-F": {'dist': (0.478, 0.01), 'l2rel': (0.4, 0.086), 'norefine': (0.48, 0.05), 'annealed': (0.45, 0.01)},
    },
    "5x10": {
        "MLGD": {'dist': (0.08, 0.012), 'l2rel': (0.043, 0.007), 'norefine': (0.22, 0.06), 'oracle': (0.18, 0.08)},
        "MLGD-F": {'dist': (0.252, 0.017), 'l2rel': (0.746, 0.289), 'norefine': (0.42, 0.09)},
        "PMLGD": {'dist': (0.088, 0.02), 'l2rel': (0.126, 0.126), 'norefine': (0.18, 0.04), 'annealed': (0.18, 0.04), 'oracle': (0.12, 0.03)},
        "PMLGD-F": {'dist': (0.276, 0.017), 'l2rel': (0.892, 0.179), 'norefine': (0.41, 0.04), 'annealed': (0.41, 0.04)},
    },
    "10x1": {
        "MLGD": {'dist': (2.234, 0.947), 'l2rel': (0.952, 0.331), 'norefine': (0.59, 0.24), 'oracle': (0.89, 0.1)},
        "MLGD-F": {'dist': (1.802, 0.445), 'l2rel': (1.237, 0.187), 'norefine': (0.61, 0.08)},
        "PMLGD": {'dist': (3.231, 0.822), 'l2rel': (2.81, 1.047), 'norefine': (0.95, 0.09), 'annealed': (0.76, 0.29), 'oracle': (0.84, 0.11)},
        "PMLGD-F": {'dist': (2.113, 0.625), 'l2rel': (1.708, 0.626), 'norefine': (0.75, 0.25), 'annealed': (0.48, 0.11)},
    },
    "10x5": {
        "MLGD": {'dist': (0.587, 0.275), 'l2rel': (0.233, 0.122), 'norefine': (0.31, 0.07), 'oracle': (0.54, 0.15)},
        "MLGD-F": {'dist': (1.102, 0.255), 'l2rel': (0.876, 0.049), 'norefine': (0.43, 0.1)},
        "PMLGD": {'dist': (1.041, 0.546), 'l2rel': (0.393, 0.347), 'norefine': (0.55, 0.18), 'annealed': (0.29, 0.06), 'oracle': (0.63, 0.1)},
        "PMLGD-F": {'dist': (1.172, 0.415), 'l2rel': (0.855, 0.268), 'norefine': (0.52, 0.08), 'annealed': (0.4, 0.06)},
    },
    "10x10": {
        "MLGD": {'dist': (0.227, 0.041), 'l2rel': (0.168, 0.1), 'norefine': (0.66, 0.03), 'oracle': (0.54, 0.2)},
        "MLGD-F": {'dist': (1.321, 0.008), 'l2rel': (2.92, 0.079), 'norefine': (0.83, 0.11)},
        "PMLGD": {'dist': (0.188, 0.026), 'l2rel': (1.063, 0.579), 'norefine': (0.8, 0.1), 'annealed': (0.62, 0.13), 'oracle': (0.72, 0.27)},
        "PMLGD-F": {'dist': (1.288, 0.04), 'l2rel': (2.469, 0.502), 'norefine': (0.97, 0.02), 'annealed': (0.93, 0.06)},
    },
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
