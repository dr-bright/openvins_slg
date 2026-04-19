#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Fill these in:
#   WEIGHT_FILES: list of artifact filenames attached to the release
# -----------------------------------------------------------------------------

WEIGHT_FILES=(
  "weights_latest/superpoint.onnx"
  "weights_latest/superpoint_lightglue_fused_cpu.onnx"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ONNX_DIR="${REPO_ROOT}/onnx"

raw_remote_url="$(git -C "${REPO_ROOT}" remote get-url origin)"
if [[ -z "${raw_remote_url}" ]]; then
  echo "Failed to get origin URL from git remote." >&2
  exit 1
fi

REPO_URL="${raw_remote_url}"
# Convert SSH form to HTTPS if needed:
#   git@github.com:org/repo.git -> https://github.com/org/repo
if [[ "${REPO_URL}" =~ ^git@([^:]+):(.*)$ ]]; then
  REPO_URL="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
fi
# Remove trailing .git for release URL composition.
REPO_URL="${REPO_URL%.git}"

if [[ "${#WEIGHT_FILES[@]}" -eq 0 ]]; then
  echo "No WEIGHT_FILES specified. Add filenames to WEIGHT_FILES array in scripts/fetch_weights.sh." >&2
  exit 1
fi

mkdir -p "${ONNX_DIR}"

download_file() {
  local file_name="$1"
  local url="${REPO_URL}/releases/download/${file_name}"
  local out_path="${ONNX_DIR}/${file_name}"

  echo "Downloading ${file_name}"
  curl --create-dirs -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "${out_path}" "${url}"

  if [[ ! -s "${out_path}" ]]; then
    echo "Downloaded file is empty: ${out_path}" >&2
    return 1
  fi
}

for f in "${WEIGHT_FILES[@]}"; do
  download_file "${f}"
done

echo "Done. Weights saved to: ${ONNX_DIR}"
