#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source plugins/detection_ayq/scripts/export_local_runtime_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_AYQ_ROOT="${PLUGIN_DIR}/ayq_runtime"

if [[ ! -d "${LOCAL_AYQ_ROOT}/mmdet" ]]; then
  echo "[detection_ayq] Missing local runtime at: ${LOCAL_AYQ_ROOT}/mmdet" >&2
  return 1 2>/dev/null || exit 1
fi

export AYQ_ROOT="${LOCAL_AYQ_ROOT}"
export AYQ_PYTHON_EXECUTABLE="${AYQ_PYTHON_EXECUTABLE:-/home/bryanswkim/anaconda3/envs/ayq/bin/python}"

echo "[detection_ayq] AYQ_ROOT=${AYQ_ROOT}"
echo "[detection_ayq] AYQ_PYTHON_EXECUTABLE=${AYQ_PYTHON_EXECUTABLE}"
echo "[detection_ayq] Environment exported for ChatClinic backend."
