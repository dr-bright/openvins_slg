#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="${SCRIPT_DIR}/../.."

# source "${REPO_ROOT}/scripts/unit_tests/_common.sh"

BAG_PATH="$(realpath "$1")"
BAG_DIR="$(dirname "${BAG_PATH}")"
BAG_NAME="$(basename "${BAG_PATH}")"
TEST_ARTIFACT_DIR="${BAG_DIR}/ov_lightglue_tests"
CONFIG="${BAG_DIR}/../config/openvins_slg/gopro1440_klt.yaml"
EST_OUT="${TEST_ARTIFACT_DIR}/traj_klt.txt"
TIME_OUT="${TEST_ARTIFACT_DIR}/timing_klt.txt"
EVAL_OUT="${TEST_ARTIFACT_DIR}/metrics_klt.txt"
MP4_OUT="${TEST_ARTIFACT_DIR}/viz_klt.mp4"

mkdir -p "${TEST_ARTIFACT_DIR}"

roslaunch ov_lightglue serial.launch \
  config:=gopro10 \
  dataset:="${BAG_NAME}" \
  bag:="${BAG_PATH}" \
  config_path:="${CONFIG}" \
  dosave:=true \
  dotime:=false \
  doeval:=true \
  doviz:=true \
  path_est:="${EST_OUT}" \
  path_time:="${TIME_OUT}" \
  path_metrics:="${EVAL_OUT}" \
  path_vis:="${MP4_OUT}" \
  slg_use_gpu:=true

