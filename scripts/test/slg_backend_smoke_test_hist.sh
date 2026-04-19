#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

prepare_artifacts "slg_backend_smoke_test_hist"
resolve_models
check_binary "slg_backend_smoke_test_hist"
# extract_two_frames_from_bag "${IMAGE_TOPIC_DEFAULT}"
FRAME0="/data/gopro10/_first/frames/vlcsnap-2026-02-24-03h07m07s971.png"
FRAME1="/data/gopro10/_first/frames/vlcsnap-2026-02-24-03h07m19s095.png"

run_and_log "${TEST_ARTIFACT_DIR}/slg_backend_smoke_test_hist.log" \
  "${BIN_DIR}/slg_backend_smoke_test_hist" \
  "${SUPERPOINT_MODEL}" "${LIGHTGLUE_MODEL}" "${FRAME0}" "${FRAME1}" 1

