#!/usr/bin/env bash
set -euo pipefail
sim_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  "$sim_root/.venv-mujoco-hardware/bin/python" "$sim_root/scripts/sync_mujoco_home.py" "$@"
