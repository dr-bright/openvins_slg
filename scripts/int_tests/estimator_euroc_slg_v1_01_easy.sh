#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="${SCRIPT_DIR}/../.."

# source "${REPO_ROOT}/scripts/unit_tests/_common.sh"

BAG_PATH="/data/euroc_mav/V1_01_easy/V1_01_easy.bag"
BAG_DIR="$(dirname "${BAG_PATH}")"
TEST_ARTIFACT_DIR="${BAG_DIR}/ov_lightglue_tests"
CONFIG_SLG="${REPO_ROOT}/config/euroc_mav/estimator_config_slg.yaml"
EST_OUT="${TEST_ARTIFACT_DIR}/traj_slg.txt"
TIME_OUT="${TEST_ARTIFACT_DIR}/timing_slg.txt"
# LOG_OUT="${TEST_ARTIFACT_DIR}/roslaunch.log"

mkdir -p "${TEST_ARTIFACT_DIR}"

roslaunch ov_lightglue serial.launch \
  config:=euroc_mav \
  dataset:=V1_01_easy \
  bag:="${BAG_PATH}" \
  config_path:="${CONFIG_SLG}" \
  dosave:=true \
  dotime:=false \
  path_est:="${EST_OUT}" \
  path_time:="${TIME_OUT}" \
  slg_use_gpu:=true

