"""Put the repo root and the mnist/ directory on sys.path (no packaging config)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(MNIST_DIR, "..", "..", ".."))
for _p in (REPO_ROOT, MNIST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
