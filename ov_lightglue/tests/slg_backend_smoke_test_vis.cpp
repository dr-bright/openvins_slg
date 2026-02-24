/*
 * Standalone visualization smoke test for ov_lightglue slg_backend.
 */

#include "track/slg_backend.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct VisState {
  const std::vector<cv::KeyPoint> *kpts0 = nullptr;
  const std::vector<cv::KeyPoint> *kpts1 = nullptr;
  const std::vector<cv::DMatch> *matches = nullptr;
  int left_width = 0;
  int selected_match = -1;
};

int find_match_near_click(const VisState &state, int x, int y, float max_dist_px) {
  if (!state.kpts0 || !state.kpts1 || !state.matches) {
    return -1;
  }
  const float max_dist_sq = max_dist_px * max_dist_px;
  float best_dist_sq = std::numeric_limits<float>::max();
  int best_idx = -1;

  for (size_t i = 0; i < state.matches->size(); ++i) {
    const cv::DMatch &m = state.matches->at(i);
    if (m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(state.kpts0->size()) ||
        m.trainIdx >= static_cast<int>(state.kpts1->size())) {
      continue;
    }
    const cv::Point2f p0 = state.kpts0->at(static_cast<size_t>(m.queryIdx)).pt;
    cv::Point2f p1 = state.kpts1->at(static_cast<size_t>(m.trainIdx)).pt;
    p1.x += static_cast<float>(state.left_width);

    const float dx0 = static_cast<float>(x) - p0.x;
    const float dy0 = static_cast<float>(y) - p0.y;
    const float d0 = dx0 * dx0 + dy0 * dy0;

    const float dx1 = static_cast<float>(x) - p1.x;
    const float dy1 = static_cast<float>(y) - p1.y;
    const float d1 = dx1 * dx1 + dy1 * dy1;

    const float d = std::min(d0, d1);
    if (d <= max_dist_sq && d < best_dist_sq) {
      best_dist_sq = d;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

void on_mouse(int event, int x, int y, int /*flags*/, void *userdata) {
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) {
    return;
  }
  VisState *state = reinterpret_cast<VisState *>(userdata);
  const int idx = find_match_near_click(*state, x, y, 8.0f);
  if (idx >= 0) {
    state->selected_match = idx;
  }
}

cv::Mat draw_side_by_side(const cv::Mat &img0, const cv::Mat &img1, const std::vector<cv::KeyPoint> &kpts0,
                          const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::DMatch> &matches, bool show_matches, int selected_match) {
  cv::Mat left;
  cv::Mat right;
  if (img0.channels() == 1) {
    cv::cvtColor(img0, left, cv::COLOR_GRAY2BGR);
  } else {
    left = img0.clone();
  }
  if (img1.channels() == 1) {
    cv::cvtColor(img1, right, cv::COLOR_GRAY2BGR);
  } else {
    right = img1.clone();
  }

  const int out_h = std::max(left.rows, right.rows);
  const int out_w = left.cols + right.cols;
  cv::Mat canvas(out_h, out_w, CV_8UC3, cv::Scalar(20, 20, 20));
  left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
  right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));

  const cv::Scalar red(0, 0, 255);
  for (const cv::KeyPoint &kp : kpts0) {
    cv::circle(canvas, kp.pt, 2, red, cv::FILLED, cv::LINE_AA);
  }
  for (const cv::KeyPoint &kp : kpts1) {
    cv::Point2f p = kp.pt;
    p.x += static_cast<float>(left.cols);
    cv::circle(canvas, p, 2, red, cv::FILLED, cv::LINE_AA);
  }

  if (show_matches) {
    const cv::Scalar green(0, 255, 0);
    for (const cv::DMatch &m : matches) {
      if (m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(kpts0.size()) ||
          m.trainIdx >= static_cast<int>(kpts1.size())) {
        continue;
      }
      cv::Point2f p0 = kpts0[static_cast<size_t>(m.queryIdx)].pt;
      cv::Point2f p1 = kpts1[static_cast<size_t>(m.trainIdx)].pt;
      p1.x += static_cast<float>(left.cols);
      cv::line(canvas, p0, p1, green, 1, cv::LINE_AA);
    }
  }

  if (!show_matches && selected_match >= 0 && selected_match < static_cast<int>(matches.size())) {
    const cv::DMatch &m = matches[static_cast<size_t>(selected_match)];
    if (m.queryIdx >= 0 && m.trainIdx >= 0 && m.queryIdx < static_cast<int>(kpts0.size()) && m.trainIdx < static_cast<int>(kpts1.size())) {
      const cv::Scalar green(0, 255, 0);
      cv::Point2f p0 = kpts0[static_cast<size_t>(m.queryIdx)].pt;
      cv::Point2f p1 = kpts1[static_cast<size_t>(m.trainIdx)].pt;
      p1.x += static_cast<float>(left.cols);
      cv::line(canvas, p0, p1, green, 2, cv::LINE_AA);
      cv::circle(canvas, p0, 4, green, cv::FILLED, cv::LINE_AA);
      cv::circle(canvas, p1, 4, green, cv::FILLED, cv::LINE_AA);
    }
  }

  return canvas;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <superpoint.onnx> <lightglue.onnx> <image0> <image1> [use_gpu=1]" << std::endl;
    return 2;
  }

  const std::string superpoint_onnx_path = argv[1];
  const std::string lightglue_onnx_path = argv[2];
  const bool use_gpu = (argc >= 6) ? (std::atoi(argv[5]) != 0) : true;

  const cv::Mat img0 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
  const cv::Mat img1 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
  if (img0.empty() || img1.empty()) {
    std::cerr << "Failed to load input images." << std::endl;
    return 3;
  }

  try {
    ov_lightglue::slg_backend backend(superpoint_onnx_path, lightglue_onnx_path, use_gpu, ov_lightglue::slg_backend::log_level::info);

    std::vector<cv::KeyPoint> kpts0;
    std::vector<cv::KeyPoint> kpts1;
    cv::Mat desc0;
    cv::Mat desc1;
    backend.run_superpoint(img0, kpts0, desc0, 1024, -1.0f);
    backend.run_superpoint(img1, kpts1, desc1, 1024, -1.0f);

    std::vector<cv::DMatch> matches;
    backend.run_lightglue(img0.size(), kpts0, desc0, img1.size(), kpts1, desc1, matches, -1.0f);

    std::cout << "kpts0=" << kpts0.size() << " kpts1=" << kpts1.size() << " matches=" << matches.size() << std::endl;
    std::cout << "Controls: SPACE toggle all matches, mouse click near matched keypoint shows that pair, q/ESC quit" << std::endl;

    bool show_matches = false;

    VisState vis_state;
    vis_state.kpts0 = &kpts0;
    vis_state.kpts1 = &kpts1;
    vis_state.matches = &matches;
    vis_state.left_width = img0.cols;
    vis_state.selected_match = -1;

    cv::namedWindow("slg_backend_smoke_test_vis", cv::WINDOW_NORMAL);
    cv::setMouseCallback("slg_backend_smoke_test_vis", on_mouse, &vis_state);

    while (true) {
      cv::Mat canvas = draw_side_by_side(img0, img1, kpts0, kpts1, matches, show_matches, vis_state.selected_match);
      cv::imshow("slg_backend_smoke_test_vis", canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
      }
      if (key == ' ') {
        show_matches = !show_matches;
      }
    }

    cv::destroyAllWindows();
  } catch (const std::exception &e) {
    std::cerr << "slg_backend visualization smoke test failed: " << e.what() << std::endl;
    return 4;
  }

  return 0;
}
