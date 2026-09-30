"""Tier 0.5: the Stable Diffusion experiment's CPU-checkable parts.

No GPU, no model download, seconds. What it establishes:
  - the ported package's imports resolve and every module byte-compiles, and its MMD is
    the shared compute_mmd (the two checks `run.sh check` runs);
  - the N = 2,000 evaluation's bookkeeping (scribble discovery in both layouts, the
    binomial CI and delta% against values from the paper's own eval_all_results.json,
    the table) -- everything of it that is not generation;
  - the interpolation figures' axes (orientation, monotonicity in age) and the sorted
    grid's even / by-hand picks;
  - export_results.py on a fake finished run with baselines/ and eval/.

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
    assert list(ev.find_scribbles(str(paper), [])) == ["source", "avg", "sdedit_best", "mlgd_f"]


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
    for f in ["scribble.png", "final_scribble_mlgd_f.png", "baselines/scribble_sdedit_best.png",
              "baselines/preview_mlgd_f.png", "eval/preview_source.png"]:
        Image.new("RGB", (4, 4)).save(run / f)
    json.dump({"n_eval": 2000, "results": {"source": {"mmd": .565}, "mlgd_f": {"mmd": .386}}},
              open(run / "eval" / "eval_results.json", "w"))
    (run / "eval" / "eval_table.md").write_text("# eval: N = 2000\n\n| Method | MMD |\n|---|---|\n| mlgd_f | 0.386 |\n")
    (run / "eval" / "embeddings.npz").write_bytes(b"big")   # must not be exported

    r = subprocess.run([sys.executable, os.path.join(SD_EXP, "export_results.py"),
                        "--run-dir", str(outs), "--dest", str(tmp_path / "res"),
                        "--interp-dir", str(tmp_path / "none"), "--interp-dest", str(tmp_path / "ires")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    dest = tmp_path / "res" / "full_BalancedTarget_seed1_123"
    assert (dest / "steps.csv").exists() and (dest / "curves.png").exists()
    assert (dest / "baselines" / "scribble_sdedit_best.png").exists()
    assert (dest / "eval" / "eval_results.json").exists() and not (dest / "eval" / "embeddings.npz").exists()
    summary = (tmp_path / "res" / "mlgdf_runs.md").read_text()
    assert "| full_BalancedTarget_seed1_123 | BalancedTarget | 1 | 123 | 0.386 | 0.565 |" in summary
    assert "## full_BalancedTarget_seed1_123: N = 2000" in (tmp_path / "res" / "eval_table.md").read_text()
