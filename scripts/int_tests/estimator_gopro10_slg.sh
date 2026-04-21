#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="${SCRIPT_DIR}/../.."

# source "${REPO_ROOT}/scripts/unit_tests/_common.sh"

BAG_PATH="$(realpath "$1")"
BAG_DIR="$(dirname "${BAG_PATH}")"
BAG_NAME="$(basename "${BAG_PATH}")"
TEST_ARTIFACT_DIR="${BAG_DIR}/ov_lightglue_tests"
CONFIG="${BAG_DIR}/../config/openvins_slg/gopro1440_slg.yaml"
EST_OUT="${TEST_ARTIFACT_DIR}/traj_slg.txt"
TIME_OUT="${TEST_ARTIFACT_DIR}/timing_slg.txt"
EVAL_OUT="${TEST_ARTIFACT_DIR}/metrics_slg.txt"

mkdir -p "${TEST_ARTIFACT_DIR}"

roslaunch ov_lightglue serial.launch \
  config:=gopro10 \
  dataset:="${BAG_NAME}" \
  bag:="${BAG_PATH}" \
  config_path:="${CONFIG}" \
  dosave:=true \
  dotime:=false \
  doeval:=true \
  path_est:="${EST_OUT}" \
  path_time:="${TIME_OUT}" \
  path_metrics:="${EVAL_OUT}" \
  slg_use_gpu:=true

