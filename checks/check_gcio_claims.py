#!/usr/bin/env python
"""Keep the README's `gcio` story and the code in agreement.

`gcio.results.save_run` is the run store's write entry point. The README's
"Results" section makes a *checkable* claim about who calls it: exactly which
entry points outside `gcio/` and `tests/` persist their runs. That claim has
been wrong before (it used to say tables are built from the store; they are
not), so this script pins it.

It fails in BOTH directions, which is the point:

  * someone wires `save_run` into another script and does not touch the README
    -> the derived caller set grows, the README's set does not, FAIL;
  * someone edits the README to claim a caller that does not exist (or drops a
    real one) -> the sets diverge the other way, FAIL.

The README carries a machine-readable list in an HTML comment:

    <!-- gcio-callers: scripts/a.py scripts/b.py -->

and the script additionally requires each of those paths, and the spelled-out
count, to appear in the visible prose -- so the comment cannot be quietly
"fixed" while the paragraph a reader actually sees stays wrong.

Usage:  python checks/check_gcio_claims.py [repo_root]
Exit 0 = README and code agree.  Exit 1 = they do not.
"""

import ast
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
             "node_modules", "notebooks", "legacy"}
# Directories whose callers do not count as "entry points using the store":
# gcio/ is the store itself, tests/ exercises it with a temp store, checks/ is us.
EXEMPT_PREFIXES = ("gcio/", "tests/", "checks/")

MARKER = re.compile(r"<!--\s*gcio-callers:\s*(.*?)\s*-->", re.S)
NUMBER_WORD = {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
               6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def actual_callers(root):
    """Every .py file outside the exempt dirs containing a call to save_run().

    AST, not grep: a mention in a docstring or a comment is not a caller.
    """
    hits = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if rel.startswith(EXEMPT_PREFIXES):
                continue
            try:
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            except SyntaxError as exc:
                print(f"SYNTAX ERROR {rel}: {exc}")
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else None)
                if name == "save_run":
                    hits.setdefault(rel, []).append(node.lineno)
    return hits


def claimed_callers(readme_text):
    m = MARKER.search(readme_text)
    if m is None:
        return None
    return sorted(set(m.group(1).split()))


def main(root="."):
    root = os.path.abspath(root)
    readme_path = os.path.join(root, "README.md")
    readme = open(readme_path, encoding="utf-8").read()

    claimed = claimed_callers(readme)
    if claimed is None:
        print("FAIL: README.md has no '<!-- gcio-callers: ... -->' marker.\n"
              "      Add one listing the entry points that call gcio.results.save_run.")
        return 1

    hits = actual_callers(root)
    actual = sorted(hits)

    problems = []
    for path in sorted(set(actual) - set(claimed)):
        problems.append(f"  calls save_run but the README does not list it: {path} "
                        f"(line(s) {', '.join(map(str, hits[path]))})")
    for path in sorted(set(claimed) - set(actual)):
        problems.append(f"  the README lists it but it does not call save_run: {path}")

    # The hidden marker must also match what a human reads.
    body = MARKER.sub("", readme)
    for path in claimed:
        if f"`{path}`" not in body:
            problems.append(f"  listed in the marker but never mentioned in the README "
                            f"prose as `{path}`")
    word = NUMBER_WORD.get(len(actual))
    if word is not None and f"{word} entry point" not in body:
        problems.append(f"  the README prose does not say \"{word} entry point"
                        f"{'' if len(actual) == 1 else 's'} call `save_run`\" "
                        f"({len(actual)} do)")

    if problems:
        print("FAIL: README.md and the code disagree about who calls gcio.results.save_run.")
        print("\n".join(problems))
        print(f"\n  README claims ({len(claimed)}): {', '.join(claimed) or '(none)'}")
        print(f"  code has      ({len(actual)}): {', '.join(actual) or '(none)'}")
        return 1

    print(f"OK: {len(actual)} entry point(s) call gcio.results.save_run, "
          f"and README.md says so:")
    for path in actual:
        print(f"  {path}: line(s) {', '.join(map(str, hits[path]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
