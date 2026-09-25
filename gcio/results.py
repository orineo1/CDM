"""Persistence for optimization runs.

The single most consequential gap found in the audit was that nothing in this
repo ever wrote results to disk. That is why the synthetic results table in
The original analysis notebook consisted of ~150 hand-pasted
`torch.tensor(...)` literals: with nothing persisted, someone had to copy
console output by hand, and the published table cannot be regenerated.

Every run should call `save_run(...)`. Tables are then built by reading the
store, never by pasting numbers into a notebook cell.

Format: one JSON object per line (JSONL), appended. Safe for concurrent
appends of small records on a local filesystem, greppable, and diff-friendly.
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

DEFAULT_STORE = os.environ.get(
    "GC_RESULTS_STORE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "runs.jsonl"),
)


def _git_hash() -> Optional[str]:
    """Short hash of the working tree, so a result can be traced to code."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        h = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def _jsonable(x: Any) -> Any:
    """Best-effort conversion of tensors / dataclasses / arrays to JSON."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "detach") and hasattr(x, "tolist"):      # torch.Tensor
        return x.detach().cpu().tolist()
    if hasattr(x, "tolist"):                                # numpy
        return x.tolist()
    if callable(x):
        return getattr(x, "__name__", repr(x))
    return repr(x)


def save_run(experiment: str,
             config: Any,
             metrics: dict,
             seed: Optional[int] = None,
             store: Optional[str] = None,
             **extra) -> dict:
    """Append one run record to the store and return it.

    Args:
        experiment: e.g. "synthetic_2d", "mnist_bimodal", "equivalence"
        config:     GuidanceConfig, dataclass, or plain dict
        metrics:    {"final_loss": ..., "l2_to_optimal": ..., "time_s": ...}
        seed:       the run's seed, if any
        store:      override the JSONL path
    """
    if hasattr(config, "describe"):
        cfg = config.describe()
    else:
        cfg = _jsonable(config)

    record = {
        "experiment": experiment,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_hash(),
        "host": platform.node(),
        "seed": seed,
        "config": cfg,
        "metrics": _jsonable(metrics),
    }
    record.update(_jsonable(extra))

    path = store or DEFAULT_STORE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_runs(experiment: Optional[str] = None,
              store: Optional[str] = None) -> list:
    """Read records back, optionally filtered by experiment name."""
    path = store or DEFAULT_STORE
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if experiment is None or rec.get("experiment") == experiment:
                out.append(rec)
    return out


def to_dataframe(experiment: Optional[str] = None, store: Optional[str] = None):
    """Flatten records into a pandas DataFrame for table building.

    This is what notebooks should call instead of hardcoding numbers.
    """
    import pandas as pd

    rows = []
    for rec in load_runs(experiment, store):
        row = {
            "experiment": rec.get("experiment"),
            "timestamp": rec.get("timestamp"),
            "git": rec.get("git"),
            "seed": rec.get("seed"),
        }
        for k, v in (rec.get("metrics") or {}).items():
            row[f"metric.{k}"] = v
        for k, v in (rec.get("config") or {}).items():
            row[f"cfg.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
