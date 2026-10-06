# Python environment of every slurm script (each sources this file after `cd` to the repository root).
#
# CDM_ENV: auto | old | new | /path/to/venv
#   old   CDM_venv        Debian 12, Python 3.11 (the cluster before its upgrade of 2026-10-06)
#   new   CDM_venv_deb13  Debian 13, Python 3.13
#   auto  old if its Python loads numpy and torch on this node, new otherwise
# Set it here for all scripts, or per job: CDM_ENV=new sbatch ...   A VENV=/path/to/venv given to sbatch
# overrides both.
CDM_ENV=${CDM_ENV:-auto}

# Site paths (lab storage for venvs and caches) come from cluster_local.sh, which is not in git:
#   CDM_LAB=/path/to/lab/storage   (holds venvs/ and hf_cache/)
[ -f ./cluster_local.sh ] && . ./cluster_local.sh
_venvs=${CDM_LAB:-.}/venvs
case "$CDM_ENV" in
  old)  _venv=$_venvs/CDM_venv ;;
  new)  _venv=$_venvs/CDM_venv_deb13 ;;
  auto) if "$_venvs/CDM_venv/bin/python" -c "import numpy, torch" >/dev/null 2>&1; then _venv=$_venvs/CDM_venv
        else _venv=$_venvs/CDM_venv_deb13; fi ;;
  *)    _venv=$CDM_ENV ;;
esac
VENV=${VENV:-$_venv}
if [ -f "$VENV/bin/activate" ]; then
  . "$VENV/bin/activate"
else
  echo "cluster_env.sh: no virtual environment at $VENV (CDM_ENV=$CDM_ENV)" >&2
fi

# GPU driver libraries. Since the Debian 13 upgrade (2026-10-06) the NVIDIA user-space driver is not installed
# system-wide but in /usr/local/nvidia/<version> (the `nvidia` modules point there); use the version that
# matches the node's kernel module.
if [ -r /proc/driver/nvidia/version ] && ! ldconfig -p 2>/dev/null | grep -q "libcuda.so"; then
  _nv=$(sed -n 's/.*Kernel Module[^0-9]*\([0-9][0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -1)
  [ -d "/usr/local/nvidia/$_nv/lib" ] || _nv=$(ls /usr/local/nvidia 2>/dev/null | grep -E '^[0-9]' | sort -V | tail -1)
  if [ -n "$_nv" ] && [ -d "/usr/local/nvidia/$_nv/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/nvidia/$_nv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PATH="/usr/local/nvidia/$_nv/bin:$PATH"
  fi
fi
