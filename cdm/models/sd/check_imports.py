"""
Static integrity check for the SDXL package. Needs NO third-party SD
dependency and no GPU, so it is runnable anywhere.

It verifies that every intra-package import (`from cdm.models.sd.metrics import ...`)
resolves. Every module is parsed with `ast`; for each
`from cdm.models.sd.X import a, b` we confirm `cdm/models/sd/X.py` exists and defines `a` and `b` at
top level, and that no bare sibling import (`from image_utils import ...`) is left.
Also byte-compiles every file.

    python -m cdm.models.sd.check_imports
"""
import ast
import pathlib
import py_compile
import sys

PKG = pathlib.Path(__file__).parent


def toplevel_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


def main():
    files = sorted(p for p in PKG.glob("*.py") if p.name != "__init__.py")
    trees, errors = {}, []

    for p in files:
        try:
            trees[p.stem] = ast.parse(p.read_text())
        except SyntaxError as e:
            errors.append(f"SYNTAX {p.name}: {e}")

    exported = {stem: toplevel_names(t) for stem, t in trees.items()}

    for stem, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cdm.models.sd."):
                target = node.module[len("cdm.models.sd."):]
                if target not in exported:
                    errors.append(f"{stem}.py -> no such module cdm.models.sd.{target}")
                    continue
                for a in node.names:
                    if a.name not in exported[target]:
                        errors.append(f"{stem}.py -> cdm.models.sd.{target} does not define '{a.name}'")

        # bare sibling imports (`from metrics import ...`) only work with the
        # package directory on sys.path, which it is not when run as `python -m`
        for node in ast.walk(tree):
            mods = ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                    else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for m in mods:
                if m.split(".")[0] in trees:
                    errors.append(f"{stem}.py -> bare sibling import '{m}' "
                                  f"(use cdm.models.sd.{m})")

    for p in files:
        try:
            py_compile.compile(str(p), doraise=True, cfile=str(PKG / "__pycache__" / (p.stem + ".chk")))
        except py_compile.PyCompileError as e:
            errors.append(f"COMPILE {p.name}: {e}")

    print(f"modules parsed : {len(trees)}")
    n_edges = sum(
        1 for t in trees.values() for n in ast.walk(t)
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("cdm.models.sd.")
    )
    print(f"intra-package imports checked : {n_edges}")
    for e in errors:
        print("  ERROR", e)
    print("\nPASS" if not errors else f"\nFAIL ({len(errors)})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
