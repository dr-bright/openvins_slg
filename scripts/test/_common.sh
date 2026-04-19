#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WS_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"

BAG_PATH="/data/gopro10/slow_fast/720.bag"
IMAGE_TOPIC_DEFAULT="/cam0/image_raw"
ARTIFACT_ROOT="/data/gopro10/slow_fast/ov_lightglue_tests"
BIN_DIR="${WS_ROOT}/devel/.private/ov_lightglue/lib/ov_lightglue"
ONNX_DIR="${REPO_ROOT}/onnx"

SP_MODEL_DEFAULT="${ONNX_DIR}/weights_latest/superpoint.onnx"
LG_MODEL_DEFAULT="${ONNX_DIR}/weights_latest/superpoint_lightglue_fused_cpu.onnx"

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "Missing file: ${p}" >&2
    exit 1
  fi
}

resolve_models() {
  SUPERPOINT_MODEL="${SP_MODEL_DEFAULT}"
  LIGHTGLUE_MODEL="${LG_MODEL_DEFAULT}"

  if [[ ! -d "${ONNX_DIR}" ]]; then
    echo "Missing ONNX directory: ${ONNX_DIR}" >&2
    echo "Fetch weights first: ${REPO_ROOT}/scripts/fetch_weights.sh" >&2
    exit 1
  fi

  if [[ ! -f "${SUPERPOINT_MODEL}" ]]; then
    SUPERPOINT_MODEL="$(find "${ONNX_DIR}" -type f \( -name 'superpoint*.onnx' -o -name '*superpoint*.onnx' \) | sort | head -n1 || true)"
  fi
  if [[ ! -f "${LIGHTGLUE_MODEL}" ]]; then
    LIGHTGLUE_MODEL="$(find "${ONNX_DIR}" -type f \( -name 'superglue*.onnx' -o -name '*lightglue*.onnx' -o -name '*glue*.onnx' \) | sort | head -n1 || true)"
  fi

  require_file "${SUPERPOINT_MODEL}"
  require_file "${LIGHTGLUE_MODEL}"
}

prepare_artifacts() {
  local test_name="$1"
  TEST_ARTIFACT_DIR="${ARTIFACT_ROOT}/${test_name}"
  mkdir -p "${TEST_ARTIFACT_DIR}"
}

extract_two_frames_from_bag() {
  local topic="${1:-${IMAGE_TOPIC_DEFAULT}}"
  FRAME0="${TEST_ARTIFACT_DIR}/frame_0000.png"
  FRAME1="${TEST_ARTIFACT_DIR}/frame_0001.png"
  if [[ -f "${FRAME0}" && -f "${FRAME1}" ]]; then
    return 0
  fi

  python3 - "${BAG_PATH}" "${topic}" "${FRAME0}" "${FRAME1}" <<'PY'
import sys
import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bag_path, topic, out0, out1 = sys.argv[1:5]
bridge = CvBridge()
count = 0

with rosbag.Bag(bag_path, "r") as bag:
    for _, msg, _ in bag.read_messages(topics=[topic]):
        if not isinstance(msg, Image):
            continue
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        out = out0 if count == 0 else out1
        cv2.imwrite(out, img)
        count += 1
        if count >= 2:
            break

if count < 2:
    raise RuntimeError(f"Could not extract 2 images from topic {topic}")
PY
}

run_and_log() {
  local log_path="$1"
  shift
  echo "Running: $*" | tee "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

check_binary() {
  local bin="$1"
  if [[ ! -x "${BIN_DIR}/${bin}" ]]; then
    echo "Binary not found: ${BIN_DIR}/${bin}" >&2
    echo "Build first: cd ${WS_ROOT} && catkin build ov_lightglue -j4 --no-status" >&2
    exit 1
  fi
}
