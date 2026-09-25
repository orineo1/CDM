"""Regression guards for the reorganisation items. Tier 0.5, CPU, seconds.

These are deliberately NOT general coverage. Each test pins one defect that the
reorg fixed, and each was demonstrated to FAIL on the pre-fix tree:

  * ``test_consistency_model_builds_from_utils_nn``  -- Item 4. Fails against
    ``git show 6abf831:models/samplers/consistency.py``, where two locally
    redefined classes shadowed the ``utils.nn`` import and ``ConsistencyModel``
    silently built the wrong (smaller) network.
  * ``test_no_imported_name_is_redefined``           -- Items 1 and 6. The AST
    sweep that found the Item 4 bug, promoted from ``checks/ast_shadow_sweep.py``
    so the bug class cannot come back. 2 intersections pre-Item-4, 0 now.
  * ``test_likelihood_imports_as_a_package`` and
    ``test_flow_matching_resolves_to_site_packages`` -- Item 6. The two library
    modules moved out of ``scripts/`` into ``likelihood/``; the second test is
    the name-collision trap, because ``likelihood/optimizer.py`` imports the PIP
    ``flow_matching`` package and this repo also ships ``models/samplers/
    flow_matching.py``.

Note what the equivalence gate structurally cannot do here: it compares two arms
that import the SAME modules, so none of these defects would ever redden it.
See ``tests/README.md``.
"""

import ast
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# --------------------------------------------------------------------------
# Item 4: ConsistencyModel must build its network from utils.nn
# --------------------------------------------------------------------------

def test_consistency_model_builds_from_utils_nn():
    """The network is resolved at runtime, so ask the object, not the source.

    Pre-Item-4 this returned "models.samplers.consistency" (the shadowing local
    class) instead of "utils.nn".
    """
    from models.samplers.consistency import ConsistencyModel

    model = ConsistencyModel(nfeatures=1, condition_on=1, nunits=8, depth=2)
    assert type(model.net).__module__ == "utils.nn", (
        f"ConsistencyModel.net is {type(model.net).__module__}."
        f"{type(model.net).__name__}; expected the utils.nn class. A local "
        f"redefinition in models/samplers/consistency.py is shadowing the import "
        f"again -- see Item 4."
    )


def test_ict_model_keeps_its_own_network():
    """The iCT variant deliberately uses the private iCT-specific network.

    The Item 4 fix renamed the shadowing classes rather than deleting them, so
    this pins that ConsistencyModeliCT still gets its own, and the rename did
    not quietly repoint it at utils.nn.
    """
    consistency = pytest.importorskip("models.samplers.consistency")
    if not hasattr(consistency, "ConsistencyModeliCT"):
        pytest.skip("no ConsistencyModeliCT in this tree")
    model = consistency.ConsistencyModeliCT(nfeatures=1, condition_on=1,
                                            nunits=8, depth=2)
    assert type(model.net).__module__ == "models.samplers.consistency"


# --------------------------------------------------------------------------
# Items 1 + 6: no module may import a name and then redefine it
# --------------------------------------------------------------------------

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
             "node_modules"}


def _shadowing_intersections(root):
    """(path, name, import_line, def_line) for every imported-then-redefined name.

    Python resolves module globals at call time, so a top-level definition wins
    over an import of the same name *regardless of order in the file* -- which
    is exactly how the Item 4 bug hid in plain sight.
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
    # legacy/ is a byte-for-byte freeze of pre-refactor code: it is not ours to fix.
    hits = [h for h in hits if not h[0].startswith("legacy/")]
    assert hits == [], "imported name redefined at module level:\n" + "\n".join(
        f"  {p}: '{n}' imported line {il}, redefined line {dl}"
        for p, n, il, dl in hits)


# --------------------------------------------------------------------------
# Item 6: the moved libraries, and the flow_matching collision trap
# --------------------------------------------------------------------------

def test_likelihood_imports_as_a_package():
    import likelihood

    assert likelihood.__file__.endswith(os.path.join("likelihood", "__init__.py"))
    for name in ("optimize_likelihood", "optimize_likelihood_direct",
                 "get_representative_points", "get_noised_representatives"):
        assert callable(getattr(likelihood, name)), f"likelihood.{name} missing"


def test_flow_matching_resolves_to_site_packages():
    """likelihood/optimizer.py wants the PIP flow_matching, not ours.

    This repo ships models/samplers/flow_matching.py. If anything ever puts that
    directory on sys.path, or a top-level flow_matching.py appears, the import in
    likelihood/optimizer.py silently binds to the wrong module.
    """
    flow_matching = pytest.importorskip(
        "flow_matching", reason="pip flow_matching not installed in this env")
    path = os.path.abspath(flow_matching.__file__)
    assert "site-packages" in path or "dist-packages" in path, (
        f"flow_matching resolved to {path}, not the installed package")
    assert not path.startswith(REPO + os.sep), (
        f"flow_matching resolved INSIDE this repo ({path}) -- the repo's own "
        f"models/samplers/flow_matching.py is shadowing the pip package")
