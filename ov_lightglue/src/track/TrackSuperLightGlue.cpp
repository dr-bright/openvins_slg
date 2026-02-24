/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 *
 * Author: drbright <gkigki111@gmail.com>
 */

#include "track/TrackSuperLightGlue.h"
#include "track/slg_backend.h"

#include "cam/CamBase.h"
#include "feat/FeatureDatabase.h"

#include <utility>

namespace ov_lightglue {

TrackSuperLightGlue::TrackSuperLightGlue(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco,
                                         bool stereo, HistogramMethod histmethod, slg_config config)
    : TrackBase(std::move(cameras), numfeats, numaruco, stereo, histmethod), config_(std::move(config)) {
  initialize_models();
}

TrackSuperLightGlue::~TrackSuperLightGlue() = default;

void TrackSuperLightGlue::feed_new_camera(const ov_core::CameraData &message) {
  if (message.sensor_ids.empty()) {
    return;
  }

  // if (message.sensor_ids.size() == 1) {
  //   feed_monocular(message, 0);
  //   return;
  // }

  // if (use_stereo && message.sensor_ids.size() >= 2) {
  //   feed_stereo(message, 0, 1);
  //   return;
  // }

  for (size_t i = 0; i < message.sensor_ids.size(); i++) {
    feed_monocular(message, i);
  }
}

void TrackSuperLightGlue::feed_monocular(const ov_core::CameraData &message, size_t msg_id) {
  if (msg_id >= message.images.size() || msg_id >= message.sensor_ids.size()) {
    return;
  }
  (void)message;
  // TODO: run SuperPoint on current image, match to previous frame, filter, and update database.
}

void TrackSuperLightGlue::feed_stereo(const ov_core::CameraData &message, size_t msg_id_left, size_t msg_id_right) {
  (void)message;
  (void)msg_id_left;
  (void)msg_id_right;
}

void TrackSuperLightGlue::initialize_models() {
  backend_.reset(new slg_backend(config_));
}

void TrackSuperLightGlue::run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) {
  if (img.empty()) {
    kpts.clear();
    desc.release();
    return;
  }
  backend_->run_superpoint(img, kpts, desc);
}

void TrackSuperLightGlue::run_lightglue(const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0,
                                        const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1,
                                        std::vector<cv::DMatch> &matches) {
  backend_->run_lightglue(kpts0, desc0, kpts1, desc1, matches);
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

} // namespace ov_lightglue
