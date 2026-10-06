# Python environment of every slurm script (each sources this file after `cd` to the repository root).
#
# VENV: a virtual environment with CUDA torch (default .venv in the repository root). Set it here for
# all scripts, or per job: VENV=/path/to/venv sbatch ...
VENV=${VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then
  . "$VENV/bin/activate"
else
  echo "cluster_env.sh: no virtual environment at $VENV" >&2
fi
