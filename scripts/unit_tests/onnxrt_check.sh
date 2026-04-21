#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

prepare_artifacts "onnxrt_check"
check_binary "onnxrt_check"
run_and_log "${TEST_ARTIFACT_DIR}/onnxrt_check.log" "${BIN_DIR}/onnxrt_check" info

