/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 */

#ifndef OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H
#define OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace ov_lightglue {

class slg_backend {
public:
  enum class log_level {
    verbose = 0,
    info = 1,
    warning = 2,
    error = 3,
    fatal = 4,
  };

  /**
   * @brief Construct and initialize ONNX Runtime sessions for SuperPoint and LightGlue.
   *
   * This sets up execution providers and model sessions once so repeated frame processing
   * can reuse initialized runtime resources with minimal overhead.
   */
  explicit slg_backend(const std::string &superpoint_onnx_path, const std::string &lightglue_onnx_path, bool use_gpu = true,
                       log_level verbosity = log_level::warning);
  ~slg_backend();

  slg_backend(const slg_backend &) = delete;
  slg_backend &operator=(const slg_backend &) = delete;
  slg_backend(slg_backend &&) noexcept;
  slg_backend &operator=(slg_backend &&) noexcept;

  /**
   * @brief Run SuperPoint on one image and return keypoints with descriptors.
   *
   * This isolates feature extraction so tracker logic can request features without handling
   * ONNX tensor plumbing or model-specific output formatting.
   */
  void run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc, int max_keypoints = 1024,
                      float min_confidence = 0.5f) const;
  /**
   * @brief Run LightGlue matching between two keypoint/descriptor sets.
   *
   * This performs model input normalization, batched inference, and conversion into OpenCV
   * match objects so downstream code can use a standard match interface.
   */
  void run_lightglue(const cv::Size &size0, const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, const cv::Size &size1,
                     const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1, std::vector<cv::DMatch> &matches,
                     float min_confidence = 0.5f, bool keypoints_normalized = false) const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H
