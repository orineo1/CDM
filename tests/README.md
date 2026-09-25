# tests/

Four tiers, cheapest first. `scripts/run_all_tests.sh` runs Tiers 0-2.

| file | tier | what it establishes |
|---|---|---|
| `test_smoke.py` | 0 | the packages import and one tiny run completes |
| `test_equivalence.py` | 1 | the refactored optimizers match `legacy/Optimization_legacy.py` on fixed seeds — **the merge gate** |
| `test_determinism.py` | 2 | the same seed twice gives the same answer |
| `test_population.py` | — | particle/refinement plumbing, structure only (the file says outright that its values are meaningless) |
| `test_refactor_guards.py` | 0.5 | one pinned defect per reorganisation item (added by this PR) |
| `scripts/run_regression.py` | 3 | *intended* to compare against published numbers — **see below, it does not currently work** |

All of the above are CPU-only, seconds, tiny untrained models, no downloads.

## Read this before trusting a green run

### Tier 1 proves agreement, not correctness

`test_equivalence.py` compares the refactored path against the frozen
`legacy/Optimization_legacy.py`. But **both arms import the same modules under
`models/`, `losses/` and `utils/`.** A defect in a shared dependency is present
identically on both sides, the difference cancels, and the gate stays green.

Item 4 of the reorganisation is the worked example. `models/samplers/consistency.py`
imported `GenericNN` and `TimeEmbedding` from `utils.nn` and then redefined both
further down the file with smaller, different implementations; Python resolves
module globals at call time, so `ConsistencyModel` built the *local* network and
the import was dead. The equivalence gate exercises that exact code — it imports
`ConsistencyModel`, instantiates it, and runs with `CM=True` — and was green
throughout. After the fix both arms shifted by exactly the same amount and the
gate reported `rel_diff=0.0000%`.

So: a green Tier 1 says the two implementations agree. It says nothing about
whether they are right. Defects of the "both sides are wrong in the same way"
shape need a test that inspects the object rather than diffing two arms —
`test_refactor_guards.py` is that kind of test.

### Tier 3 is not currently a backstop either

The docs point at `scripts/run_regression.py` as the check that catches what
Tier 1 cannot, because it compares against externally recorded published numbers.
Two things make that untrue today (verified 2026-09-25 at `ba71795`):

1. **There is no baseline.** `tests/golden/` is empty and untracked
   (`git ls-files tests/golden` returns nothing), so a plain
   `python scripts/run_regression.py` runs the five seeds and then exits **2**
   with `No baseline at .../tests/golden/synthetic_2d.json`. A fresh clone has
   nothing to compare against, and nothing in CI creates one.
2. **The models it scores are random inits.** `build_models()` constructs
   `DiffusionModel` and `ConsistencyModel` under a fixed seed and loads no
   weights — there is no `load_state_dict`, `torch.load` or `from_pretrained`
   anywhere in the file. Even with a golden file, what it pins is the behaviour
   of untrained networks on a fixed seed, which is a determinism check wearing a
   regression check's clothes. It is not the paper's numbers.

Until a baseline is committed and the scored models are the trained ones, treat
Tier 3 as a stub, not as coverage.

## Adding tests here

Anything in `tests/` runs in CI on every push, so keep new tests CPU-only,
seconds, deterministic and download-free. `test_refactor_guards.py` is
deliberately narrow: one guard per fixed defect, each demonstrated to fail on the
pre-fix tree, not a general expansion of coverage.
