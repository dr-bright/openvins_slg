/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 *
 * Author: drbright <gkigki111@gmail.com>
 */

#ifndef OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H
#define OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H

#include <memory>
#include <string>
#include "track/TrackBase.h"

namespace ov_lightglue {

class slg_backend;

/**
 * @brief Runtime configuration for SuperPoint + LightGlue backend.
 *
 * The image size fields are used only for LightGlue keypoint normalization.
 * If either dimension is left as zero, normalization factors are estimated
 * from keypoint coordinates at runtime, which is slightly slower.
 */
struct slg_config {
  /// Absolute path to SuperPoint ONNX model.
  std::string superpoint_onnx_path;
  /// Absolute path to LightGlue ONNX model.
  std::string lightglue_onnx_path;
  /// Hard cap applied after SuperPoint scoring to limit downstream cost.
  int max_keypoints = 1024;
  /// Input image width used for LightGlue keypoint normalization.
  int input_width = 0;
  /// Input image height used for LightGlue keypoint normalization.
  int input_height = 0;
  /// Request CUDA execution provider when available.
  bool use_gpu = true;
};

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
   */
  explicit TrackSuperLightGlue(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco,
                               bool stereo, HistogramMethod histmethod, slg_config config);

  ~TrackSuperLightGlue() override;

  /**
   * @brief Process new camera measurements.
   */
  void feed_new_camera(const ov_core::CameraData &message) override;

protected:
  /**
   * @brief Monocular processing entry-point.
   */
  void feed_monocular(const ov_core::CameraData &message, size_t msg_id);

  /**
   * @brief Stereo/binocular processing entry-point.
   */
  void feed_stereo(const ov_core::CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Initialize SuperPoint and LightGlue inference backends.
   */
  void initialize_models();

  /**
   * @brief Run SuperPoint extraction on a single image.
   */
  void run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

  /**
   * @brief Run LightGlue matching between descriptor sets.
   */
  void run_lightglue(const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, const std::vector<cv::KeyPoint> &kpts1,
                     const cv::Mat &desc1, std::vector<cv::DMatch> &matches);

  /**
   * @brief Push filtered feature observations to the shared FeatureDatabase.
   */
  void update_feature_database(double timestamp, size_t cam_id, const std::vector<cv::KeyPoint> &kpts,
                               const std::vector<size_t> &ids);

private:
  std::unique_ptr<slg_backend> backend_;
  slg_config config_;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_TRACK_SUPER_LIGHTGLUE_H
