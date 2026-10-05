"""Narrow regression guards. CPU, seconds.

  * ``test_consistency_model_builds_from_utils_nn``: a locally redefined class must not
    shadow the ``cdm.utils.nn`` import, or ``ConsistencyModel`` silently builds the wrong
    (smaller) network.
  * ``test_no_imported_name_is_redefined``: an AST sweep of the repository for any module
    that imports a name and then redefines it at top level (the same bug class).
"""

import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# ConsistencyModel must build its network from utils.nn
# --------------------------------------------------------------------------

def test_consistency_model_builds_from_utils_nn():
    """The network is resolved at runtime, so ask the object, not the source."""
    from cdm.models.samplers.consistency import ConsistencyModel

    model = ConsistencyModel(nfeatures=1, condition_on=1, nunits=8, depth=2)
    assert type(model.net).__module__ == "cdm.utils.nn", (
        f"ConsistencyModel.net is {type(model.net).__module__}."
        f"{type(model.net).__name__}; expected the cdm.utils.nn class. A local "
        f"redefinition in cdm/models/samplers/consistency.py is shadowing the import "
        f"again."
    )


def test_ict_model_keeps_its_own_network():
    """The iCT variant deliberately uses the private iCT-specific network.

    This pins that ConsistencyModeliCT still gets its own network, not the
    cdm.utils.nn one.
    """
    from cdm.models.samplers import consistency
    model = consistency.ConsistencyModeliCT(nfeatures=1, condition_on=1,
                                            nunits=8, depth=2)
    assert type(model.net).__module__ == "cdm.models.samplers.consistency"


# --------------------------------------------------------------------------
# No module may import a name and then redefine it
# --------------------------------------------------------------------------

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
             "node_modules"}


def _shadowing_intersections(root):
    """(path, name, import_line, def_line) for every imported-then-redefined name.

    Python resolves module globals at call time, so a top-level definition wins
    over an import of the same name *regardless of order in the file* -- which
    is how such a bug hides in plain sight.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            except SyntaxError:
                continue            # not our business; other tooling catches it
            imported = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.setdefault(
                            alias.asname or alias.name.split(".")[0], node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != "*":
                            imported.setdefault(alias.asname or alias.name, node.lineno)
            defined = {}
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    defined.setdefault(node.name, node.lineno)
            for name in sorted(set(imported) & set(defined)):
                out.append((rel, name, imported[name], defined[name]))
    return out


def test_no_imported_name_is_redefined():
    hits = _shadowing_intersections(REPO)
    assert hits == [], "imported name redefined at module level:\n" + "\n".join(
        f"  {p}: '{n}' imported line {il}, redefined line {dl}"
        for p, n, il, dl in hits)

