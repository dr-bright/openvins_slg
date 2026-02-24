/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 */

#ifndef OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H
#define OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H

#if !__has_include(<onnxruntime_cxx_api.h>)
#error "ONNX Runtime headers not found. Install ONNX Runtime and add its include path to build ov_lightglue."
#endif
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace ov_lightglue {

struct slg_config;

class slg_backend {
public:
  /**
   * @brief Construct and initialize ONNX Runtime sessions for SuperPoint and LightGlue.
   *
   * This sets up execution providers and model sessions once so repeated frame processing
   * can reuse initialized runtime resources with minimal overhead.
   */
  explicit slg_backend(const slg_config &config, OrtLoggingLevel verbosity = ORT_LOGGING_LEVEL_WARNING);

  /**
   * @brief Run SuperPoint on one image and return keypoints with descriptors.
   *
   * This isolates feature extraction so tracker logic can request features without handling
   * ONNX tensor plumbing or model-specific output formatting.
   */
  void run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) const;
  /**
   * @brief Run LightGlue matching between two keypoint/descriptor sets.
   *
   * This performs model input normalization, batched inference, and conversion into OpenCV
   * match objects so downstream code can use a standard match interface.
   */
  void run_lightglue(const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, const std::vector<cv::KeyPoint> &kpts1,
                     const cv::Mat &desc1, std::vector<cv::DMatch> &matches) const;

private:
  /**
   * @brief Convert input image to float format expected by SuperPoint.
   *
   * Normalizing and channel-converting here keeps model input consistent and avoids repeated
   * preprocessing logic in higher-level tracking code.
   */
  static cv::Mat preprocess_image(const cv::Mat &img, bool force_gray);
  /**
   * @brief Normalize keypoint coordinates to LightGlue's canonical coordinate range.
   *
   * LightGlue expects normalized coordinates rather than raw pixels, so this transformation
   * is required before building matcher input tensors.
   */
  static std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::KeyPoint> &kpts, int h, int w);

  /**
   * @brief Read input node names from an ONNX Runtime session.
   *
   * Caching names avoids allocator work on every inference call and ensures stable
   * input ordering for model execution.
   */
  static std::vector<std::string> get_input_names(const Ort::Session &session);
  /**
   * @brief Read output node names from an ONNX Runtime session.
   *
   * Output names are needed by ONNX Runtime Run() and are cached once for lower overhead
   * during repeated inference.
   */
  static std::vector<std::string> get_output_names(const Ort::Session &session);
  /**
   * @brief Read input tensor shapes from an ONNX Runtime session.
   *
   * Baseline model shapes are captured once so per-frame code can adjust only dynamic
   * dimensions instead of re-querying metadata repeatedly.
   */
  static std::vector<std::vector<int64_t>> get_input_shapes(const Ort::Session &session);

  int input_width_ = 0;
  int input_height_ = 0;
  int max_keypoints_ = 1024;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> superpoint_session_;
  std::unique_ptr<Ort::Session> lightglue_session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;

  std::vector<std::string> superpoint_input_names_;
  std::vector<std::string> superpoint_output_names_;
  std::vector<std::vector<int64_t>> superpoint_input_shapes_;

  std::vector<std::string> lightglue_input_names_;
  std::vector<std::string> lightglue_output_names_;
};

} // namespace ov_lightglue

#endif // OV_LIGHTGLUE_TRACK_INFERENCE_BACKEND_H
