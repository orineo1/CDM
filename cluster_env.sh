# Python environment of every slurm script (each sources this file after `cd` to the repository root).
#
# CDM_ENV: auto | old | new | /path/to/venv
#   old   CDM_venv        Debian 12, Python 3.11 (the cluster before its upgrade of 2026-10-06)
#   new   CDM_venv_deb13  Debian 13, Python 3.13
#   auto  new on Debian 13 nodes, old otherwise
# Set it here for all scripts, or per job: CDM_ENV=new sbatch ...   A VENV=/path/to/venv given to sbatch
# overrides both.
CDM_ENV=${CDM_ENV:-auto}

_venvs=/sci/labs/orzuk/orzuk/venvs
case "$CDM_ENV" in
  old)  _venv=$_venvs/CDM_venv ;;
  new)  _venv=$_venvs/CDM_venv_deb13 ;;
  auto) if grep -q "^13" /etc/debian_version 2>/dev/null; then _venv=$_venvs/CDM_venv_deb13; else _venv=$_venvs/CDM_venv; fi ;;
  *)    _venv=$CDM_ENV ;;
esac
VENV=${VENV:-$_venv}
if [ -f "$VENV/bin/activate" ]; then
  . "$VENV/bin/activate"
else
  echo "cluster_env.sh: no virtual environment at $VENV (CDM_ENV=$CDM_ENV)" >&2
fi
