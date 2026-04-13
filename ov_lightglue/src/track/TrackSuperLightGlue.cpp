/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 *
 * Author: drbright <gkigki111@gmail.com>
 */

#include "track/TrackSuperLightGlue.h"
#include "track/slg_backend.h"

#include "cam/CamBase.h"
#include "feat/FeatureDatabase.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace ov_lightglue {

TrackSuperLightGlue::TrackSuperLightGlue(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco,
                                         bool stereo, HistogramMethod histmethod, TrackSuperLightGlueConfig config)
    : TrackBase(std::move(cameras), numfeats, numaruco, stereo, histmethod), config_(std::move(config)) {
  initialize_models();
}

TrackSuperLightGlue::~TrackSuperLightGlue() = default;

void TrackSuperLightGlue::feed_new_camera(const ov_core::CameraData &message) {
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    throw std::runtime_error("TrackSuperLightGlue: CameraData sizes do not match");
  }

  if (message.images.size() == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
    return;
  }

  for (size_t i = 0; i < message.sensor_ids.size(); ++i) {
    feed_monocular(message, i);
  }
}

void TrackSuperLightGlue::feed_monocular(const ov_core::CameraData &message, size_t msg_id) {
  if (msg_id >= message.images.size() || msg_id >= message.sensor_ids.size() || msg_id >= message.masks.size()) {
    return;
  }

  const size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  const cv::Mat img = preprocess_image(message.images.at(msg_id));
  const cv::Mat mask = message.masks.at(msg_id);

  std::vector<cv::KeyPoint> curr_kpts;
  cv::Mat curr_desc;
  run_superpoint(img, curr_kpts, curr_desc);

  if (curr_kpts.empty() || curr_desc.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    desc_last_[cam_id].release();
    return;
  }

  const bool has_prev = (pts_last.find(cam_id) != pts_last.end() && !pts_last.at(cam_id).empty() && desc_last_.find(cam_id) != desc_last_.end() &&
                         !desc_last_.at(cam_id).empty());

  std::vector<cv::DMatch> temporal_matches;
  if (has_prev) {
    const std::vector<cv::KeyPoint> prev_kpts_undist = undistort_keypoints(cam_id, pts_last.at(cam_id));
    const std::vector<cv::KeyPoint> curr_kpts_undist = undistort_keypoints(cam_id, curr_kpts);
    run_lightglue(img_last.at(cam_id).size(), prev_kpts_undist, desc_last_.at(cam_id), img.size(), curr_kpts_undist, curr_desc, temporal_matches,
                  true);
    temporal_matches = filter_temporal_matches(temporal_matches, pts_last.at(cam_id), curr_kpts, img.size(), mask);
    temporal_matches = apply_temporal_ransac(temporal_matches, prev_kpts_undist, curr_kpts_undist);
  }

  std::vector<cv::KeyPoint> accepted_kpts;
  std::vector<size_t> accepted_ids;
  cv::Mat accepted_desc;
  accepted_kpts.reserve(static_cast<size_t>(num_features));
  accepted_ids.reserve(static_cast<size_t>(num_features));

  std::set<int> used_curr_indices;
  for (const cv::DMatch &m : temporal_matches) {
    if (static_cast<int>(accepted_kpts.size()) >= num_features) {
      break;
    }
    if (m.queryIdx < 0 || m.trainIdx < 0 || m.trainIdx >= static_cast<int>(curr_kpts.size()) ||
        m.queryIdx >= static_cast<int>(pts_last.at(cam_id).size()) || used_curr_indices.count(m.trainIdx) > 0) {
      continue;
    }
    const cv::KeyPoint &kp = curr_kpts.at(static_cast<size_t>(m.trainIdx));
    if (!is_keypoint_usable(kp, img.size(), mask) || !passes_spatial_filter(kp, accepted_kpts)) {
      continue;
    }

    accepted_kpts.push_back(kp);
    accepted_ids.push_back(ids_last.at(cam_id).at(static_cast<size_t>(m.queryIdx)));
    append_descriptor_row(curr_desc, m.trainIdx, accepted_desc);
    used_curr_indices.insert(m.trainIdx);
  }

  if (config_.refill_tracks) {
    for (int idx = 0; idx < curr_desc.rows; ++idx) {
      if (static_cast<int>(accepted_kpts.size()) >= num_features) {
        break;
      }
      if (used_curr_indices.count(idx) > 0) {
        continue;
      }

      const cv::KeyPoint &kp = curr_kpts.at(static_cast<size_t>(idx));
      if (!is_keypoint_usable(kp, img.size(), mask) || !passes_spatial_filter(kp, accepted_kpts)) {
        continue;
      }

      accepted_kpts.push_back(kp);
      accepted_ids.push_back(++currid);
      append_descriptor_row(curr_desc, idx, accepted_desc);
    }
  }

  update_feature_database(message.timestamp, cam_id, accepted_kpts, accepted_ids);

  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = accepted_kpts;
    ids_last[cam_id] = accepted_ids;
    desc_last_[cam_id] = accepted_desc;
  }
}

void TrackSuperLightGlue::feed_stereo(const ov_core::CameraData &message, size_t msg_id_left, size_t msg_id_right) {
  feed_monocular(message, msg_id_left);
  feed_monocular(message, msg_id_right);
}

void TrackSuperLightGlue::initialize_models() {
  backend_.reset(new slg_backend(config_.superpoint_onnx_path, config_.lightglue_onnx_path, config_.use_gpu));
}

void TrackSuperLightGlue::run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) {
  if (img.empty()) {
    kpts.clear();
    desc.release();
    return;
  }
  backend_->run_superpoint(img, kpts, desc, config_.max_keypoints, config_.detect_min_confidence);
}

void TrackSuperLightGlue::run_lightglue(const cv::Size &size0, const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0,
                                        const cv::Size &size1, const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1,
                                        std::vector<cv::DMatch> &matches, bool keypoints_normalized) {
  backend_->run_lightglue(size0, kpts0, desc0, size1, kpts1, desc1, matches, config_.match_min_confidence, keypoints_normalized);
}

void TrackSuperLightGlue::update_feature_database(double timestamp, size_t cam_id, const std::vector<cv::KeyPoint> &kpts,
                                                  const std::vector<size_t> &ids) {
  if (kpts.size() != ids.size()) {
    return;
  }

  for (size_t i = 0; i < kpts.size(); i++) {
    const cv::Point2f uv = kpts.at(i).pt;
    const cv::Point2f uvn = camera_calib.at(cam_id)->undistort_cv(uv);
    database->update_feature(ids.at(i), timestamp, cam_id, uv.x, uv.y, uvn.x, uvn.y);
  }
}

cv::Mat TrackSuperLightGlue::preprocess_image(const cv::Mat &img) const {
  if (img.empty()) {
    return img;
  }

  cv::Mat gray = img;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat out;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(gray, out);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
    clahe->apply(gray, out);
  } else {
    out = gray;
  }
  return out;
}

std::vector<cv::DMatch> TrackSuperLightGlue::filter_temporal_matches(const std::vector<cv::DMatch> &raw_matches,
                                                                      const std::vector<cv::KeyPoint> &prev_kpts,
                                                                      const std::vector<cv::KeyPoint> &curr_kpts,
                                                                      const cv::Size &image_size,
                                                                      const cv::Mat &curr_mask) const {
  std::vector<cv::DMatch> ordered = raw_matches;
  std::sort(ordered.begin(), ordered.end(), [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });

  std::vector<cv::DMatch> filtered;
  filtered.reserve(ordered.size());
  std::set<int> used_query;
  std::set<int> used_train;

  for (const cv::DMatch &m : ordered) {
    if (m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(prev_kpts.size()) || m.trainIdx >= static_cast<int>(curr_kpts.size())) {
      continue;
    }
    if (used_query.count(m.queryIdx) > 0 || used_train.count(m.trainIdx) > 0) {
      continue;
    }

    if (!is_keypoint_usable(curr_kpts.at(static_cast<size_t>(m.trainIdx)), image_size, curr_mask)) {
      continue;
    }

    filtered.push_back(m);
    used_query.insert(m.queryIdx);
    used_train.insert(m.trainIdx);
  }

  return filtered;
}

std::vector<cv::DMatch> TrackSuperLightGlue::apply_temporal_ransac(const std::vector<cv::DMatch> &matches,
                                                                    const std::vector<cv::KeyPoint> &prev_kpts,
                                                                    const std::vector<cv::KeyPoint> &curr_kpts) const {
  if (!config_.enable_temporal_ransac || static_cast<int>(matches.size()) < config_.temporal_ransac_min_matches) {
    return matches;
  }

  std::vector<cv::Point2f> pts_prev;
  std::vector<cv::Point2f> pts_curr;
  pts_prev.reserve(matches.size());
  pts_curr.reserve(matches.size());
  for (const cv::DMatch &m : matches) {
    pts_prev.push_back(prev_kpts.at(static_cast<size_t>(m.queryIdx)).pt);
    pts_curr.push_back(curr_kpts.at(static_cast<size_t>(m.trainIdx)).pt);
  }

  std::vector<uchar> inlier_mask;
  cv::findFundamentalMat(pts_prev, pts_curr, cv::FM_RANSAC, config_.temporal_ransac_threshold_px, config_.temporal_ransac_confidence,
                         inlier_mask);
  if (inlier_mask.size() != matches.size()) {
    return matches;
  }

  std::vector<cv::DMatch> inliers;
  inliers.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inlier_mask[i] != 0) {
      inliers.push_back(matches[i]);
    }
  }

  if (static_cast<int>(inliers.size()) < config_.temporal_ransac_min_inliers) {
    return matches;
  }
  return inliers;
}

bool TrackSuperLightGlue::is_keypoint_usable(const cv::KeyPoint &kp, const cv::Size &image_size, const cv::Mat &mask) const {
  const int x = static_cast<int>(std::round(kp.pt.x));
  const int y = static_cast<int>(std::round(kp.pt.y));
  if (x < 0 || y < 0 || x >= image_size.width || y >= image_size.height) {
    return false;
  }
  if (mask.empty()) {
    return true;
  }
  return static_cast<int>(mask.at<uint8_t>(y, x)) <= 127;
}

bool TrackSuperLightGlue::passes_spatial_filter(const cv::KeyPoint &kp, const std::vector<cv::KeyPoint> &accepted) const {
  const float min_dist = static_cast<float>(std::max(0, config_.min_feature_distance_px));
  if (min_dist <= 0.0f) {
    return true;
  }
  const float min_dist_sq = min_dist * min_dist;
  for (const cv::KeyPoint &existing : accepted) {
    const float dx = existing.pt.x - kp.pt.x;
    const float dy = existing.pt.y - kp.pt.y;
    if (dx * dx + dy * dy < min_dist_sq) {
      return false;
    }
  }
  return true;
}

std::vector<cv::KeyPoint> TrackSuperLightGlue::undistort_keypoints(size_t cam_id, const std::vector<cv::KeyPoint> &kpts) const {
  std::vector<cv::KeyPoint> out;
  out.reserve(kpts.size());
  for (const cv::KeyPoint &kp : kpts) {
    const cv::Point2f uvn = camera_calib.at(cam_id)->undistort_cv(kp.pt);
    cv::KeyPoint und = kp;
    und.pt = uvn;
    out.push_back(und);
  }
  return out;
}

void TrackSuperLightGlue::append_descriptor_row(const cv::Mat &source_desc, int row_idx, cv::Mat &out_desc) {
  if (row_idx < 0 || row_idx >= source_desc.rows) {
    return;
  }
  out_desc.push_back(source_desc.row(row_idx));
}

} // namespace ov_lightglue
