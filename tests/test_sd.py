"""The Stable Diffusion experiment's CPU-checkable parts.

No GPU, no model download, seconds. What it establishes:
  - the package's imports resolve and every module byte-compiles, and its MMD is
    the shared compute_mmd (the two checks `run.sh check` runs);
  - the N = 2,000 evaluation's bookkeeping (scribble discovery in both layouts, the
    binomial CI and delta% against values from the paper's own eval_all_results.json,
    the table) -- everything of it that is not generation;
  - the interpolation figures' axes (orientation, monotonicity in age) and the sorted
    grid's even / by-hand picks;
  - the saliency analysis' scribble change and correlations, and the step-cost
    projection and tables -- everything of them that is not a GPU pass;
  - export_results.py on a fake finished run with baselines/, eval/ and saliency/, and a
    step-cost output.

`eval_all_experiments` imports `peft` and `controlnet_aux` through the package; those are
SD-only dependencies (experiments/stable_diffusion/requirements.txt), so its part is
skipped where they are absent (e.g. the default CI environment).
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD_EXP = os.path.join(REPO, "experiments", "stable_diffusion")


def _run(script):
    return subprocess.run([sys.executable, os.path.join(REPO, "cdm", "models", "sd", script)],
                          capture_output=True, text=True, cwd=REPO)


def test_check_imports():
    r = _run("check_imports.py")
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


def test_mmd_is_the_shared_one():
    r = _run("check_mmd_equivalence.py")
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# eval_all_experiments: the pure parts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ev():
    pytest.importorskip("peft")
    pytest.importorskip("controlnet_aux")
    from cdm.models.sd import eval_all_experiments
    return eval_all_experiments


def test_eval_scribble_discovery(ev, tmp_path):
    run = tmp_path / "full_BalancedTarget_seed1_7"
    (run / "baselines").mkdir(parents=True)
    for f in ["scribble.png", "final_scribble_mlgd_f.png", "baselines/scribble_avg_prompt.png",
              "baselines/scribble_sdedit_best.png"]:
        Image.new("RGB", (4, 4)).save(run / f)
    found = ev.find_scribbles(str(run), [])
    assert list(found) == ["source", "avg_prompt", "sdedit_best", "mlgd_f"]
    paper = tmp_path / "paper"
    paper.mkdir()
    for f in ["scribble_source.png", "scribble_avg.png", "scribble_sdedit_best.png", "scribble_mlgdd.png"]:
        Image.new("RGB", (4, 4)).save(paper / f)
    assert list(ev.find_scribbles(str(paper), [])) == ["source", "sdedit_best", "mlgd_f"]   # old methods: opt-in
    both = ev.find_scribbles(str(paper), [f"avg={paper / 'scribble_avg.png'}"])
    assert list(both) == ["source", "sdedit_best", "mlgd_f", "avg"]


def test_eval_numbers_match_the_papers_json(ev):
    # SkewedTarget rows of the paper's eval_all_results.json (source vs lgd_cm)
    p, lo, hi = ev.binomial_ci(1194, 2000)
    assert abs(p - 0.597) < 1e-9 and abs(lo - 0.5755) < 1e-3 and abs(hi - 0.6185) < 1e-3
    res = {"source": {"mmd": 0.48289, "swd": 0.01271, "n": 2000},
           "mlgd_f": {"mmd": 0.35165, "swd": 0.00828, "n": 2000}}
    ev.add_deltas(res)
    assert abs(res["mlgd_f"]["mmd_improvement_pct"] - 27.2) < 0.1
    assert abs(res["mlgd_f"]["swd_improvement_pct"] - 34.85) < 0.1
    table = ev.results_table(res, "SkewedTarget", 2000, gender=False)
    assert "| mlgd_f | 0.352 | +27.2% |" in table and "p(male)" not in table


# ---------------------------------------------------------------------------
# interpolation: axes and picks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def itp():
    from cdm.models.sd import interpolation
    return interpolation


def _blob(rng, center, n, s=0.3):
    return (center + s * rng.standard_normal((n, center.shape[0]))).astype(np.float32)


def test_gender_axis_orientation_and_picks(itp, tmp_path):
    rng = np.random.default_rng(0)
    woman, man = rng.standard_normal(64), rng.standard_normal(64)
    target = {"Woman": _blob(rng, woman, 30), "WomanMasc": _blob(rng, 0.66 * woman + 0.33 * man, 30),
              "ManFem": _blob(rng, 0.33 * woman + 0.66 * man, 30), "Man": _blob(rng, man, 30)}
    axis = itp.gender_axis(target)
    assert axis.project(target["Man"])[0].mean() > axis.project(target["Woman"])[0].mean()
    embs = _blob(rng, 0.5 * (woman + man), 40, s=1.0)
    scores, coords = axis.project(embs)
    assert coords.shape == (40, 2) and np.allclose(scores, coords[:, 0])
    photos = [Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)) for _ in range(40)]
    _, picks, idx = itp.sorted_grid(photos, scores, n_picks=8, save_path=str(tmp_path / "g.png"))
    assert picks == itp.even_picks(40, 8) and all(scores[idx[i]] <= scores[idx[i + 1]] for i in range(7))
    order = np.argsort(scores)
    _, picks2, idx2 = itp.sorted_grid(photos, scores, picks=[1, 5, 9, 39], save_path=str(tmp_path / "g2.png"))
    assert picks2 == [1, 5, 9, 39] and idx2 == [int(order[r]) for r in picks2]


def test_age_axis_is_monotone_in_age(itp):
    rng = np.random.default_rng(1)
    young, old = rng.standard_normal(64), rng.standard_normal(64)
    ages = list(range(40, 80))
    target = {str(a): _blob(rng, young + (a - 40) / 39 * (old - young), 3) for a in ages}
    axis = itp.age_axis(target, 40, 79)
    means = [axis.project(target[str(a)])[0].mean() for a in ages]
    assert np.corrcoef(ages, means)[0, 1] > 0.95


# ---------------------------------------------------------------------------
# saliency and step_cost: the CPU parts
# ---------------------------------------------------------------------------

def test_saliency_change_and_correlation(tmp_path):
    from cdm.models.sd import saliency as sal
    src = np.zeros((64, 64, 3), np.float32)
    src[20:40, 10:50] = 1.0
    opt = src.copy()
    opt[20:30, 10:50] = 0.0        # a stroke removed
    opt[45:50, 5:60] = 1.0         # a stroke added
    change, signed = sal.scribble_change(src, opt, sigma=1.0)
    assert signed[47, 30] > 0 and signed[25, 30] < 0 and change[5, 5] == 0
    c = sal.change_counts(signed, 0.1)
    assert c["added_pct"] > 0 and c["removed_pct"] > 0
    assert abs(c["unchanged_pct"] + c["added_pct"] + c["removed_pct"] - 100) < 1e-9
    rng = np.random.default_rng(0)
    corr = sal.saliency_change_correlation(change + 1e-3 * rng.random(change.shape), change, src, n_bins=5)
    assert corr["n_line_pixels"] == 20 * 40 and corr["spearman_rho"] > 0.8
    assert corr["bin_saliency"] == sorted(corr["bin_saliency"]) and len(corr["bin_change"]) == 5
    lat = sal.latent_correlation(rng.random((8, 8)), rng.random((8, 8)))
    assert -1 <= lat["pearson_r"] <= 1 and sal.latent_correlation(np.ones((8, 8)), rng.random((8, 8)))["pearson_r"] is None
    sal.plot_scribble_change(src, opt, signed, 0.1, str(tmp_path / "f7.png"))
    sal.plot_saliency_correlation(change, corr, src, str(tmp_path / "f8.png"))
    assert (tmp_path / "f7.png").exists() and (tmp_path / "f8.png").exists()


def test_step_cost_projection_and_tables():
    from cdm.models.sd import step_cost as sc
    G = sc.GB
    rep = lambda t: {"t_fixed": 3.0, "t_var": t, "models": 20 * G, "fixed": 3 * G, "var": 20 * G, "peak": 43 * G}
    all100 = sc.summarize([rep(120.0), rep(122.0)], 100, 100, "all", 80 * G, 125, 15)
    assert abs(all100["step_s_mean"] - 124.0) < 1e-9 and abs(all100["step_s_std"] - 1.0) < 1e-9
    assert abs(all100["base"]["peak_gb"] - (20 + 3 + 20 * 15)) < 1e-9          # models + fixed + 15 x samples
    assert abs(all100["base"]["step_s"] - (3 + 121 * 15)) < 1e-9 and all100["base"]["x_gpu"] > 4
    assert abs(all100["run_h"] - 124 * 125 / 3600) < 1e-9
    uni100 = sc.summarize([rep(69.0)], 100, 50, "uniform", 80 * G, 125, 15)
    rows = [all100, uni100, {"rule": "all", "n_cond": 120, "n_sel": 120, "oom": True}]
    mem = sc.memory_table(rows, "A100", 125, 2, 30)
    assert "| 100 | 20.0 | 3.0 | 20.0 | 43.0 | 323 | 4.0x | 4.3 | 63 |" in mem and "| 120 | | | | OOM |" in mem
    rt = sc.runtime_table(rows, "A100", 0.5)
    assert "| 100 | 124.00 ± 1.00 | 72.00 ± 0.00 | 42% |" in rt and "| 120 | OOM | OOM |  |" in rt


# ---------------------------------------------------------------------------
# export_results on a fake finished run
# ---------------------------------------------------------------------------

def test_export_results(tmp_path):
    outs = tmp_path / "outputs"
    run = outs / "full_BalancedTarget_seed1_123"
    (run / "baselines").mkdir(parents=True)
    (run / "eval").mkdir()
    with open(run / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["step", "timestep", "mmd_loss", "gradient_norm", "zeta_i", "loss_norm",
                                "correction_norm", "elapsed_s"])
        w.writeheader()
        for i in range(3):
            w.writerow(dict(step=i + 1, timestep=1, mmd_loss=.5, gradient_norm=1, zeta_i=1, loss_norm=1,
                            correction_norm=1, elapsed_s=i))
    json.dump({"args": {"seed": 1, "num_variations": 100, "backsel_k": 50, "base_zeta": 10.0},
               "final_mlgd_f_mmd": 0.44, "final_mmd_n": 250, "optimization_time_sec": 60,
               "wandb": {"name": "x"}}, open(run / "metrics.json", "w"))
    (run / "run_info.txt").write_text("git commit: abc1234\n")
    for f in ["scribble.png", "final_scribble_mlgd_f.png", "final_photos_mlgd_f.png",
              "baselines/scribble_sdedit_best.png", "baselines/scribble_source.png",
              "baselines/preview_mlgd_f.png", "eval/preview_source.png", "eval/mmd_bar.png"]:
        Image.new("RGB", (4, 4)).save(run / f)
    json.dump({"n_eval": 2000, "results": {"source": {"mmd": .565}, "mlgd_f": {"mmd": .386}}},
              open(run / "eval" / "eval_results.json", "w"))
    (run / "eval" / "eval_table.md").write_text("# eval: N = 2000\n\n| Method | MMD |\n|---|---|\n| mlgd_f | 0.386 |\n")
    (run / "eval" / "embeddings.npz").write_bytes(b"big")   # must not be exported

    (run / "saliency").mkdir()
    json.dump({"saliency_vs_change": {"spearman_rho": 0.5}}, open(run / "saliency" / "saliency_results.json", "w"))
    for f in ["scribble_change.png", "saliency_correlation.png"]:
        Image.new("RGB", (4, 4)).save(run / "saliency" / f)
    cost = tmp_path / "cost" / "A100_7"
    cost.mkdir(parents=True)
    json.dump({"gpu": "NVIDIA A100"}, open(cost / "step_cost.json", "w"))
    (cost / "step_cost.md").write_text("# One MLGD-F guidance step\n")

    r = subprocess.run([sys.executable, os.path.join(SD_EXP, "export_results.py"),
                        "--run-dir", str(outs), "--dest", str(tmp_path / "res"),
                        "--cost-dir", str(tmp_path / "cost"), "--cost-dest", str(tmp_path / "cres")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    dest = tmp_path / "res" / "full_BalancedTarget_seed1_123"
    kept = ["metrics.json", "run_info.txt", "curves.png", "scribble_source.png", "final_scribble_mlgd_f.png",
            "baselines/scribble_sdedit_best.png", "eval/eval_results.json", "eval/preview_source.png",
            "saliency/saliency_results.json", "saliency/scribble_change.png", "saliency/saliency_correlation.png"]
    assert all((dest / f).exists() for f in kept), [f for f in kept if not (dest / f).exists()]
    # each file once: duplicates, plots of tabulated numbers and large files stay in outputs/
    dropped = ["steps.csv", "final_photos_mlgd_f.png", "baselines/scribble_source.png",
               "baselines/preview_mlgd_f.png", "eval/mmd_bar.png", "eval/eval_table.md", "eval/embeddings.npz"]
    assert not any((dest / f).exists() for f in dropped), [f for f in dropped if (dest / f).exists()]
    assert not (tmp_path / "res" / "mlgdf_runs.md").exists()
    assert "## full_BalancedTarget_seed1_123: N = 2000" in (tmp_path / "res" / "eval_table.md").read_text()
    assert (tmp_path / "cres" / "A100_7" / "step_cost.md").exists()
