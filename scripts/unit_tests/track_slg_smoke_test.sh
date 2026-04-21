#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

prepare_artifacts "track_superlightglue_smoke_test"
resolve_models
check_binary "track_superlightglue_smoke_test"
# extract_two_frames_from_bag "${IMAGE_TOPIC_DEFAULT}"
FRAME0="/data/gopro10/_first/frames/vlcsnap-2026-02-24-03h07m07s971.png"
FRAME1="/data/gopro10/_first/frames/vlcsnap-2026-02-24-03h07m19s095.png"

run_and_log "${TEST_ARTIFACT_DIR}/track_superlightglue_smoke_test.log" \
  "${BIN_DIR}/track_superlightglue_smoke_test" \
  "${SUPERPOINT_MODEL}" "${LIGHTGLUE_MODEL}" 1 "${FRAME0}" "${FRAME1}"

