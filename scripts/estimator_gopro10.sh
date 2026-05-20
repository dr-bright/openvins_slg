#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="${SCRIPT_DIR}/../.."

# source "${REPO_ROOT}/scripts/unit_tests/_common.sh"

BAG_PATH="$(realpath "$2")"
BAG_DIR="$(dirname "${BAG_PATH}")"
BAG_NAME="$(basename "${BAG_PATH}")"
TEST_ARTIFACT_DIR="${BAG_DIR}/dir_${BAG_NAME}"
TRACKER_MODE="$1"
CONFIG="${BAG_DIR}/../config/openvins_slg/gopro640_${TRACKER_MODE}.yaml"
EST_OUT="${TEST_ARTIFACT_DIR}/traj_${TRACKER_MODE}.txt"
TIME_OUT="${TEST_ARTIFACT_DIR}/timing_${TRACKER_MODE}.txt"
EVAL_OUT="${TEST_ARTIFACT_DIR}/metrics_${TRACKER_MODE}.txt"
MP4_OUT="${TEST_ARTIFACT_DIR}/viz_${TRACKER_MODE}.mp4"

BAG_START="$3"
if [ -z "${BAG_START}" ]; then
  BAG_START=0.0
fi

mkdir -p "${TEST_ARTIFACT_DIR}"

roslaunch ov_lightglue serial_new.launch \
  config:=gopro10 \
  dataset:="${BAG_NAME}" \
  bag:="${BAG_PATH}" \
  config_path:="${CONFIG}" \
  dosave:=true \
  dotime:=false \
  doeval:=false \
  doviz:=false \
  path_est:="${EST_OUT}" \
  path_time:="${TIME_OUT}" \
  path_metrics:="${EVAL_OUT}" \
  path_viz:="${MP4_OUT}" \
  slg_use_gpu:=true \
  bag_start:="${BAG_START}"

