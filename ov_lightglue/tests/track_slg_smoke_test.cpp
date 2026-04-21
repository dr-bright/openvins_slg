/*
 * Standalone smoke test for ov_lightglue TrackSuperLightGlue.
 */

#include "cam/CamRadtan.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackSuperLightGlue.h"
#include "utils/sensor_data.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

size_t count_id_overlap(const std::vector<size_t> &a, const std::vector<size_t> &b) {
  size_t ct = 0;
  for (size_t id : a) {
    if (std::find(b.begin(), b.end(), id) != b.end()) {
      ++ct;
    }
  }
  return ct;
}

cv::Mat to_bgr(const cv::Mat &img_gray) {
  cv::Mat out;
  if (img_gray.channels() == 1) {
    cv::cvtColor(img_gray, out, cv::COLOR_GRAY2BGR);
  } else {
    out = img_gray.clone();
  }
  return out;
}

struct MatchPair {
  cv::Point2f p0;
  cv::Point2f p1;
};

struct UiState {
  std::vector<MatchPair> pairs;
  float x_scale = 1.0f;
  float y_scale = 1.0f;
  int selected_pair = -1;
};

int find_pair_near(const UiState &state, float x, float y, float max_dist_px) {
  const float max_dist_sq = max_dist_px * max_dist_px;
  float best_dist_sq = std::numeric_limits<float>::max();
  int best_idx = -1;
  for (size_t i = 0; i < state.pairs.size(); ++i) {
    const MatchPair &p = state.pairs[i];
    const float dx0 = x - p.p0.x;
    const float dy0 = y - p.p0.y;
    const float d0 = dx0 * dx0 + dy0 * dy0;
    const float dx1 = x - p.p1.x;
    const float dy1 = y - p.p1.y;
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
  UiState *state = reinterpret_cast<UiState *>(userdata);
  const float cx = static_cast<float>(x) * state->x_scale;
  const float cy = static_cast<float>(y) * state->y_scale;
  const int idx = find_pair_near(*state, cx, cy, 10.0f * std::max(state->x_scale, state->y_scale));
  if (idx >= 0) {
    state->selected_pair = idx;
  }
}

cv::Mat build_transition_canvas(const cv::Mat &prev_img, const std::vector<size_t> &prev_ids, const std::vector<cv::KeyPoint> &prev_kpts,
                                const cv::Mat &curr_img, const std::vector<size_t> &curr_ids, const std::vector<cv::KeyPoint> &curr_kpts,
                                size_t frame_idx, size_t carried, size_t new_ids, size_t db_size, bool show_all, int selected_pair,
                                UiState &ui_state) {
  const int h = std::max(prev_img.rows, curr_img.rows);
  const int w = prev_img.cols + curr_img.cols;
  cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(20, 20, 20));

  cv::Mat left = to_bgr(prev_img);
  cv::Mat right = to_bgr(curr_img);
  left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
  right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));

  std::unordered_map<size_t, cv::Point2f> prev_pts_by_id;
  prev_pts_by_id.reserve(prev_ids.size());
  for (size_t i = 0; i < prev_ids.size() && i < prev_kpts.size(); ++i) {
    prev_pts_by_id[prev_ids[i]] = prev_kpts[i].pt;
  }

  std::unordered_set<size_t> carried_ids;
  carried_ids.reserve(curr_ids.size());
  ui_state.pairs.clear();
  for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
    const size_t id = curr_ids[i];
    auto it = prev_pts_by_id.find(id);
    if (it == prev_pts_by_id.end()) {
      continue;
    }
    carried_ids.insert(id);

    const cv::Point2f p0 = it->second;
    cv::Point2f p1 = curr_kpts[i].pt;
    p1.x += static_cast<float>(left.cols);
    ui_state.pairs.push_back({p0, p1});
  }

  if (show_all) {
    for (const MatchPair &p : ui_state.pairs) {
      cv::line(canvas, p.p0, p.p1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
  } else if (selected_pair >= 0 && selected_pair < static_cast<int>(ui_state.pairs.size())) {
    const MatchPair &p = ui_state.pairs[static_cast<size_t>(selected_pair)];
    cv::line(canvas, p.p0, p.p1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::circle(canvas, p.p0, 4, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, p.p1, 4, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
  }

  for (size_t i = 0; i < prev_kpts.size(); ++i) {
    cv::circle(canvas, prev_kpts[i].pt, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
  }
  for (size_t i = 0; i < curr_kpts.size() && i < curr_ids.size(); ++i) {
    cv::Point2f p = curr_kpts[i].pt;
    p.x += static_cast<float>(left.cols);
    cv::circle(canvas, p, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
  }

  const std::string title = "frame=" + std::to_string(frame_idx) + " carried=" + std::to_string(carried) + " new=" + std::to_string(new_ids) +
                            " db=" + std::to_string(db_size) + " [space:toggle lines, click:select, enter:next, q:quit]";
  cv::putText(canvas, title, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

  return canvas;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <superpoint.onnx> <lightglue.onnx> <use_gpu={0,1}> <image0> <image1> [image2 ...]" << std::endl;
    return 2;
  }

  const std::string superpoint_onnx_path = argv[1];
  const std::string lightglue_onnx_path = argv[2];
  const bool use_gpu = (std::atoi(argv[3]) != 0);
  bool enable_vis = false;
  int image_arg_end = argc;
  if (std::string(argv[argc - 1]) == "--vis") {
    enable_vis = true;
    image_arg_end = argc - 1;
  }

  std::vector<cv::Mat> images;
  images.reserve(static_cast<size_t>(std::max(0, image_arg_end - 4)));
  for (int i = 4; i < image_arg_end; ++i) {
    const cv::Mat img = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "Failed to load input image: " << argv[i] << std::endl;
      return 3;
    }
    images.push_back(img);
  }
  if (images.size() < 2) {
    std::cerr << "At least two images are required." << std::endl;
    return 3;
  }

  try {
    std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
    std::shared_ptr<ov_core::CamBase> cam = std::make_shared<ov_core::CamRadtan>(images.front().cols, images.front().rows);
    Eigen::Matrix<double, 8, 1> calib;
    calib << 300.0, 300.0, static_cast<double>(images.front().cols) * 0.5, static_cast<double>(images.front().rows) * 0.5, 0.0, 0.0, 0.0,
        0.0;
    cam->set_value(calib);
    cameras.insert({0, cam});

    ov_lightglue::TrackSuperLightGlueConfig cfg;
    cfg.superpoint_onnx_path = superpoint_onnx_path;
    cfg.lightglue_onnx_path = lightglue_onnx_path;
    cfg.use_gpu = use_gpu;
    cfg.max_keypoints = 1024;
    cfg.detect_min_confidence = -1.0f;
    cfg.match_min_confidence = -1.0f;
    cfg.enable_temporal_ransac = true;
    cfg.temporal_ransac_min_matches = 20;
    cfg.temporal_ransac_min_inliers = 12;
    cfg.temporal_ransac_threshold_px = 2.0;
    cfg.temporal_ransac_confidence = 0.99;
    cfg.min_feature_distance_px = 10;
    cfg.refill_tracks = true;

    ov_lightglue::TrackSuperLightGlue tracker(cameras, 300, 0, false, ov_core::TrackBase::HistogramMethod::NONE, cfg);

    std::cout << "TrackSuperLightGlue smoke test success\n";
    const std::string window_name = "track_superlightglue_smoke_test_vis";
    bool show_all_matches = true;
    UiState ui_state;
    if (enable_vis) {
      cv::namedWindow(window_name, cv::WINDOW_NORMAL);
      cv::setMouseCallback(window_name, on_mouse, &ui_state);
    }

    std::vector<size_t> prev_ids;
    std::vector<cv::KeyPoint> prev_kpts;
    cv::Mat prev_img;
    for (size_t frame_idx = 0; frame_idx < images.size(); ++frame_idx) {
      ov_core::CameraData msg;
      msg.timestamp = static_cast<double>(frame_idx);
      msg.sensor_ids.push_back(0);
      msg.images.push_back(images[frame_idx]);
      msg.masks.push_back(cv::Mat::zeros(images[frame_idx].rows, images[frame_idx].cols, CV_8UC1));
      tracker.feed_new_camera(msg);

      const auto ids_by_cam = tracker.get_last_ids();
      const auto obs_by_cam = tracker.get_last_obs();
      const std::vector<size_t> curr_ids = (ids_by_cam.count(0) > 0) ? ids_by_cam.at(0) : std::vector<size_t>{};
      const std::vector<cv::KeyPoint> curr_kpts = (obs_by_cam.count(0) > 0) ? obs_by_cam.at(0) : std::vector<cv::KeyPoint>{};
      const size_t active = curr_ids.size();
      const size_t carried = prev_ids.empty() ? 0 : count_id_overlap(prev_ids, curr_ids);
      const size_t new_ids = (active >= carried) ? (active - carried) : 0;

      const std::shared_ptr<ov_core::FeatureDatabase> db = tracker.get_feature_database();
      const size_t db_size = db->size();
      const size_t at_t = db->features_containing(static_cast<double>(frame_idx), false, false).size();

      std::cout << "frame=" << frame_idx << " active=" << active << " carried=" << carried << " new=" << new_ids << " db_size=" << db_size
                << " features_at_t=" << at_t << std::endl;

      if (enable_vis && frame_idx > 0) {
        ui_state.selected_pair = -1;
        while (true) {
          const cv::Mat canvas = build_transition_canvas(prev_img, prev_ids, prev_kpts, images[frame_idx], curr_ids, curr_kpts, frame_idx, carried,
                                                         new_ids, db_size, show_all_matches, ui_state.selected_pair, ui_state);

          cv::Rect win_rect;
          try {
            win_rect = cv::getWindowImageRect(window_name);
          } catch (...) {
            win_rect = cv::Rect();
          }

          cv::Mat display = canvas;
          if (win_rect.width > 0 && win_rect.height > 0 && (win_rect.width != canvas.cols || win_rect.height != canvas.rows)) {
            cv::resize(canvas, display, cv::Size(win_rect.width, win_rect.height), 0.0, 0.0, cv::INTER_LINEAR);
            ui_state.x_scale = static_cast<float>(canvas.cols) / static_cast<float>(display.cols);
            ui_state.y_scale = static_cast<float>(canvas.rows) / static_cast<float>(display.rows);
          } else {
            ui_state.x_scale = 1.0f;
            ui_state.y_scale = 1.0f;
          }

          cv::imshow(window_name, display);
          const int key = cv::waitKey(20);
          if (key == 'q' || key == 'Q' || key == 27) {
            frame_idx = images.size();
            break;
          }
          if (key == ' ') {
            show_all_matches = !show_all_matches;
          }
          if (key == 13 || key == 10) {
            break;
          }
        }
      }

      prev_ids = curr_ids;
      prev_kpts = curr_kpts;
      prev_img = images[frame_idx];
    }
    if (enable_vis) {
      cv::destroyAllWindows();
    }
  } catch (const std::exception &e) {
    std::cerr << "TrackSuperLightGlue smoke test failed: " << e.what() << std::endl;
    return 4;
  }

  return 0;
}
