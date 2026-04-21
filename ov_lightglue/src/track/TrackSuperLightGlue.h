/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 *
 * Author: drbright <gkigki111@gmail.com>
 */

#ifndef OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H
#define OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H

#include <memory>
#include <unordered_map>

#include "track/TrackBase.h"
#include "track/TrackerSuperLightGlueConfig.h"

namespace ov_lightglue {

class slg_backend;

/**
 * @brief Tracker frontend skeleton using SuperPoint features and LightGlue matching.
 *
 * This class follows the OpenVINS TrackBase contract and is intended to be
 * integrated in VioManager similarly to TrackKLT / TrackDescriptor / TrackPlane.
 */
class TrackSuperLightGlue : public ov_core::TrackBase {
public:
  /**
   * @brief Constructor with base tracker options.
   *
   * This stores tracker policy parameters and initializes backend sessions once so
   * per-frame processing can focus on feature tracking instead of model setup.
   */
  explicit TrackSuperLightGlue(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco,
                               bool stereo, HistogramMethod histmethod, TrackSuperLightGlueConfig config);

  ~TrackSuperLightGlue() override;

  /**
   * @brief Process new camera measurements.
   *
   * This is the entry point used by OpenVINS to feed synchronized image data and
   * dispatch tracking per camera while respecting tracker mode constraints.
   */
  void feed_new_camera(const ov_core::CameraData &message) override;

protected:
  /**
   * @brief Monocular processing entry-point.
   *
   * This runs extraction, temporal matching, filtering, ID propagation, and feature
   * database update for one camera image at the given timestamp.
   */
  void feed_monocular(const ov_core::CameraData &message, size_t msg_id);

  /**
   * @brief Stereo/binocular processing entry-point.
   *
   * This currently falls back to per-camera monocular updates so multi-camera
   * pipelines can run while full stereo coupling is not yet implemented.
   */
  void feed_stereo(const ov_core::CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Initialize SuperPoint and LightGlue inference backends.
   *
   * Backend model paths and execution providers are configured once here to avoid
   * repeated runtime initialization costs during image processing.
   */
  void initialize_models();

  /**
   * @brief Run SuperPoint extraction on a single image.
   *
   * This wrapper keeps tracker code independent from backend details while enabling
   * centralized handling of empty input cases and output conventions.
   */
  void run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

  /**
   * @brief Run LightGlue matching between descriptor sets.
   *
   * This wrapper centralizes model invocation so tracker logic only handles filtered
   * correspondences and ID management.
   */
  void run_lightglue(const cv::Size &size0, const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, const cv::Size &size1,
                     const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1, std::vector<cv::DMatch> &matches,
                     bool keypoints_normalized = false);

  /**
   * @brief Push filtered feature observations to the shared FeatureDatabase.
   *
   * Every accepted track must be written in both pixel and undistorted coordinates
   * so downstream VIO estimators can consume consistent measurement history.
   */
  void update_feature_database(double timestamp, size_t cam_id, const std::vector<cv::KeyPoint> &kpts,
                               const std::vector<size_t> &ids);

  /**
   * @brief Apply tracker image pre-processing configured by histogram mode.
   *
   * Matching stability depends on consistent contrast characteristics, so this keeps
   * enhancement policy in one place before feature extraction.
   */
  cv::Mat preprocess_image(const cv::Mat &img) const;

  /**
   * @brief Remove invalid, duplicate, and low-confidence temporal correspondences.
   *
   * LightGlue outputs are converted into one-to-one candidate tracks here so later
   * stages can safely propagate IDs without ambiguity.
   */
  std::vector<cv::DMatch> filter_temporal_matches(const std::vector<cv::DMatch> &raw_matches, const std::vector<cv::KeyPoint> &prev_kpts,
                                                  const std::vector<cv::KeyPoint> &curr_kpts, const cv::Size &image_size,
                                                  const cv::Mat &curr_mask) const;

  /**
   * @brief Optionally reject geometric outliers from temporal correspondences.
   *
   * RANSAC filtering suppresses inconsistent motion matches before ID carry-over to
   * reduce spurious long-lived tracks in challenging scenes.
   */
  std::vector<cv::DMatch> apply_temporal_ransac(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &prev_kpts,
                                                const std::vector<cv::KeyPoint> &curr_kpts) const;

  /**
   * @brief Check if a keypoint lies in image bounds and valid mask region.
   *
   * This prevents invalid measurements from entering the feature database and keeps
   * selection aligned with external ROI constraints supplied in camera messages.
   */
  bool is_keypoint_usable(const cv::KeyPoint &kp, const cv::Size &image_size, const cv::Mat &mask) const;

  /**
   * @brief Check if a keypoint is sufficiently far from already accepted points.
   *
   * Enforcing minimum spacing avoids over-concentrating tracks in textured patches
   * and improves spatial coverage for robust motion estimation.
   */
  bool passes_spatial_filter(const cv::KeyPoint &kp, const std::vector<cv::KeyPoint> &accepted) const;

  /**
   * @brief Convert pixel-space keypoints into undistorted normalized coordinates.
   *
   * Geometry and temporal consistency checks are more stable in normalized camera
   * coordinates, so this projection is used before LightGlue/RANSAC when requested.
   */
  std::vector<cv::KeyPoint> undistort_keypoints(size_t cam_id, const std::vector<cv::KeyPoint> &kpts) const;

  /**
   * @brief Append one descriptor row to a descriptor matrix.
   *
   * Track building gradually selects rows from dense extraction output, so this keeps
   * descriptor assembly logic compact and consistent.
   */
  static void append_descriptor_row(const cv::Mat &source_desc, int row_idx, cv::Mat &out_desc);

protected:
  std::unique_ptr<slg_backend> backend_;
  TrackSuperLightGlueConfig config_;
  std::unordered_map<size_t, cv::Mat> desc_last_;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H
