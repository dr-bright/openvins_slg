/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 */

#ifndef OV_LIGHTGLUE_TRACK_TRACKER_SUPER_LIGHTGLUE_CONFIG_H
#define OV_LIGHTGLUE_TRACK_TRACKER_SUPER_LIGHTGLUE_CONFIG_H

#include <string>

namespace ov_lightglue {

/**
 * @brief Runtime configuration for TrackSuperLightGlue.
 *
 * This struct is owned by tracker-level code and can later be populated from ROS params,
 * YAML, or other external config sources.
 */
struct TrackSuperLightGlueConfig {
  TrackSuperLightGlueConfig();

  /// Absolute path to SuperPoint ONNX model.
  std::string superpoint_onnx_path;
  /// Absolute path to LightGlue ONNX model.
  std::string lightglue_onnx_path;
  /// Hard cap applied after SuperPoint scoring to limit downstream cost.
  int max_keypoints = 1024;
  /// Keep only keypoints with confidence >= this threshold.
  float detect_min_confidence = -1.0f;
  /// Keep only matches with confidence >= this threshold.
  float match_min_confidence = -1.0f;
  /// Input image width used for LightGlue keypoint normalization.
  bool use_gpu = true;
  /// Enable RANSAC outlier rejection on temporal correspondences.
  bool enable_temporal_ransac = true;
  /// Minimum matches required before attempting temporal RANSAC.
  int temporal_ransac_min_matches = 20;
  /// Minimum inlier count required to accept temporal RANSAC filtering.
  int temporal_ransac_min_inliers = 12;
  /// RANSAC reprojection threshold in pixels.
  double temporal_ransac_threshold_px = 2.0;
  /// RANSAC confidence for geometric outlier rejection.
  double temporal_ransac_confidence = 0.99;
  /// Minimum spatial separation in pixels between accepted features.
  int min_feature_distance_px = 10;
  /// Whether to seed new tracks until target count is reached each frame.
  bool refill_tracks = true;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_TRACK_TRACKER_SUPER_LIGHTGLUE_CONFIG_H
