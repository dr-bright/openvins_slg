#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

prepare_artifacts "track_superlightglue_bag_to_mp4"
resolve_models
check_binary "track_superlightglue_bag_to_mp4"

MP4_OUT="${TEST_ARTIFACT_DIR}/slg_tracker.mp4"
CSV_OUT="${TEST_ARTIFACT_DIR}/slg_tracker_metrics.csv"

run_and_log "${TEST_ARTIFACT_DIR}/track_superlightglue_bag_to_mp4.log" \
  "${BIN_DIR}/track_superlightglue_bag_to_mp4" \
  "${SUPERPOINT_MODEL}" "${LIGHTGLUE_MODEL}" 1 "${BAG_PATH}" "${IMAGE_TOPIC_DEFAULT}" "${MP4_OUT}" "${CSV_OUT}" 20

