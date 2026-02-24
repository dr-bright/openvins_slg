/*
 * openvins_lightglue: SuperPoint + LightGlue tracking extension for OpenVINS
 */

#include "track/slg_backend.h"

#if !__has_include(<onnxruntime_cxx_api.h>)
#error "ONNX Runtime headers not found. Install ONNX Runtime and add its include path to build ov_lightglue."
#endif
#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace ov_lightglue {

namespace {

inline std::vector<const char *> to_cstr_ptrs(const std::vector<std::string> &names) {
  std::vector<const char *> out;
  out.reserve(names.size());
  for (const std::string &name : names) {
    out.push_back(name.c_str());
  }
  return out;
}

inline bool has_provider(const std::vector<std::string> &providers, const std::string &name) {
  return std::find(providers.begin(), providers.end(), name) != providers.end();
}

inline OrtLoggingLevel to_ort_log_level(slg_backend::log_level level) {
  switch (level) {
  case slg_backend::log_level::verbose:
    return ORT_LOGGING_LEVEL_VERBOSE;
  case slg_backend::log_level::info:
    return ORT_LOGGING_LEVEL_INFO;
  case slg_backend::log_level::warning:
    return ORT_LOGGING_LEVEL_WARNING;
  case slg_backend::log_level::error:
    return ORT_LOGGING_LEVEL_ERROR;
  case slg_backend::log_level::fatal:
    return ORT_LOGGING_LEVEL_FATAL;
  }
  return ORT_LOGGING_LEVEL_WARNING;
}

inline bool log_info_enabled(OrtLoggingLevel verbosity) {
  return static_cast<int>(verbosity) <= static_cast<int>(ORT_LOGGING_LEVEL_INFO);
}

inline cv::Mat preprocess_image(const cv::Mat &img, bool force_gray) {
  cv::Mat work = img;
  if (force_gray && img.channels() == 3) {
    cv::cvtColor(img, work, cv::COLOR_BGR2GRAY);
  } else if (img.channels() != 1) {
    throw std::runtime_error("slg_backend: unsupported image channels");
  }
  cv::Mat out;
  work.convertTo(out, CV_32F, 1.0 / 255.0);
  return out;
}

inline std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::KeyPoint> &kpts, int h, int w) {
  const float cx = static_cast<float>(w) * 0.5f;
  const float cy = static_cast<float>(h) * 0.5f;
  const float s = static_cast<float>(std::max(w, h)) * 0.5f;
  std::vector<cv::Point2f> out;
  out.reserve(kpts.size());
  for (const cv::KeyPoint &kp : kpts) {
    out.emplace_back((kp.pt.x - cx) / s, (kp.pt.y - cy) / s);
  }
  return out;
}

inline cv::Size infer_size_from_keypoints(const cv::Size &requested_size, const std::vector<cv::KeyPoint> &kpts) {
  if (requested_size.width > 0 && requested_size.height > 0) {
    return requested_size;
  }
  float max_x = 0.0f;
  float max_y = 0.0f;
  for (const cv::KeyPoint &kp : kpts) {
    max_x = std::max(max_x, kp.pt.x);
    max_y = std::max(max_y, kp.pt.y);
  }
  return cv::Size(std::max(1, static_cast<int>(std::ceil(max_x)) + 1), std::max(1, static_cast<int>(std::ceil(max_y)) + 1));
}

inline std::vector<std::string> get_input_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> out;
  const size_t n = session.GetInputCount();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
    out.emplace_back(name.get());
  }
  return out;
}

inline std::vector<std::string> get_output_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> out;
  const size_t n = session.GetOutputCount();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
    out.emplace_back(name.get());
  }
  return out;
}

inline std::vector<std::vector<int64_t>> get_input_shapes(const Ort::Session &session) {
  std::vector<std::vector<int64_t>> out;
  const size_t n = session.GetInputCount();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.emplace_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  return out;
}

} // namespace

struct slg_backend::impl {
  impl(const std::string &sp_path, const std::string &lg_path, bool use_gpu_requested, log_level verbosity_level)
      : superpoint_onnx_path(sp_path), lightglue_onnx_path(lg_path), use_gpu(use_gpu_requested) {
    const OrtLoggingLevel verbosity = to_ort_log_level(verbosity_level);

    env.reset(new Ort::Env(verbosity, "ov_lightglue"));
    session_options.reset(new Ort::SessionOptions());
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options->SetIntraOpNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    session_options->SetLogSeverityLevel(static_cast<int>(verbosity));

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    const bool has_cuda_ep = has_provider(providers, "CUDAExecutionProvider");
    if (use_gpu) {
      if (!has_cuda_ep) {
        if (log_info_enabled(verbosity)) {
          std::cerr << "[ov_lightglue] CUDA EP requested but not available in ONNX Runtime providers; using CPU EP." << std::endl;
        }
      } else {
        try {
          OrtCUDAProviderOptions cuda_options{};
          cuda_options.device_id = 0;
          session_options->AppendExecutionProvider_CUDA(cuda_options);
          if (log_info_enabled(verbosity)) {
            std::cerr << "[ov_lightglue] CUDA EP enabled for ONNX Runtime sessions." << std::endl;
          }
        } catch (const std::exception &e) {
          if (log_info_enabled(verbosity)) {
            std::cerr << "[ov_lightglue] CUDA EP append failed, falling back to CPU EP: " << e.what() << std::endl;
          }
        }
      }
    } else if (log_info_enabled(verbosity)) {
      std::cerr << "[ov_lightglue] GPU usage disabled by config; using CPU EP." << std::endl;
    }
    if (log_info_enabled(verbosity)) {
      std::cerr << "[ov_lightglue] Available ORT providers:";
      for (const std::string &provider : providers) {
        std::cerr << " " << provider;
      }
      std::cerr << std::endl;
    }

    superpoint_session.reset(new Ort::Session(*env, superpoint_onnx_path.c_str(), *session_options));
    lightglue_session.reset(new Ort::Session(*env, lightglue_onnx_path.c_str(), *session_options));

    memory_info.reset(new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault)));

    superpoint_input_names = get_input_names(*superpoint_session);
    superpoint_output_names = get_output_names(*superpoint_session);
    superpoint_input_shapes = get_input_shapes(*superpoint_session);

    lightglue_input_names = get_input_names(*lightglue_session);
    lightglue_output_names = get_output_names(*lightglue_session);

    if (superpoint_input_names.empty() || superpoint_output_names.size() < 3) {
      throw std::runtime_error("slg_backend: unexpected SuperPoint model IO signature");
    }
    if (lightglue_input_names.size() < 4 || lightglue_output_names.size() < 2) {
      throw std::runtime_error("slg_backend: unexpected LightGlue model IO signature");
    }
    if (log_info_enabled(verbosity)) {
      std::cerr << "[ov_lightglue] SuperPoint session initialized from: " << superpoint_onnx_path << std::endl;
      std::cerr << "[ov_lightglue] LightGlue session initialized from: " << lightglue_onnx_path << std::endl;
    }
  }

  const std::string superpoint_onnx_path;
  const std::string lightglue_onnx_path;
  const bool use_gpu = true;

  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::SessionOptions> session_options;
  std::unique_ptr<Ort::Session> superpoint_session;
  std::unique_ptr<Ort::Session> lightglue_session;
  std::unique_ptr<Ort::MemoryInfo> memory_info;

  std::vector<std::string> superpoint_input_names;
  std::vector<std::string> superpoint_output_names;
  std::vector<std::vector<int64_t>> superpoint_input_shapes;

  std::vector<std::string> lightglue_input_names;
  std::vector<std::string> lightglue_output_names;
};

slg_backend::slg_backend(const std::string &superpoint_onnx_path, const std::string &lightglue_onnx_path, bool use_gpu, log_level verbosity) {
  if (superpoint_onnx_path.empty()) {
    throw std::runtime_error("slg_backend: superpoint_onnx_path is empty");
  }
  if (lightglue_onnx_path.empty()) {
    throw std::runtime_error("slg_backend: lightglue_onnx_path is empty");
  }
  impl_.reset(new impl(superpoint_onnx_path, lightglue_onnx_path, use_gpu, verbosity));
}

slg_backend::~slg_backend() = default;

slg_backend::slg_backend(slg_backend &&) noexcept = default;
slg_backend &slg_backend::operator=(slg_backend &&) noexcept = default;

void slg_backend::run_superpoint(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc, int max_keypoints,
                                 float min_confidence) const {
  kpts.clear();
  desc.release();
  if (img.empty()) {
    return;
  }

  const cv::Mat proc = preprocess_image(img, true);
  std::vector<int64_t> input_shape = impl_->superpoint_input_shapes.front();
  if (input_shape.size() != 4) {
    throw std::runtime_error("slg_backend: SuperPoint input tensor rank must be 4");
  }
  input_shape[0] = 1;
  input_shape[1] = 1;
  input_shape[2] = proc.rows;
  input_shape[3] = proc.cols;

  std::vector<float> input_data(static_cast<size_t>(proc.rows * proc.cols));
  std::memcpy(input_data.data(), proc.ptr<float>(), sizeof(float) * input_data.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(
      Ort::Value::CreateTensor<float>(*impl_->memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));

  const std::vector<const char *> in_names = to_cstr_ptrs(impl_->superpoint_input_names);
  const std::vector<const char *> out_names = to_cstr_ptrs(impl_->superpoint_output_names);

  std::vector<Ort::Value> out_tensors = impl_->superpoint_session->Run(Ort::RunOptions{nullptr}, in_names.data(), input_tensors.data(),
                                                                        input_tensors.size(), out_names.data(), out_names.size());

  if (out_tensors.size() < 3) {
    throw std::runtime_error("slg_backend: SuperPoint output tensor count is less than 3");
  }

  const auto kpts_shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto desc_shape = out_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
  if (kpts_shape.size() != 3 || kpts_shape[2] != 2 || desc_shape.size() != 3) {
    throw std::runtime_error("slg_backend: unexpected SuperPoint output shape");
  }
  const int n = static_cast<int>(kpts_shape[1]);
  const int d = static_cast<int>(desc_shape[2]);
  if (n <= 0 || d <= 0) {
    return;
  }

  const int64_t *kpts_data = out_tensors[0].GetTensorData<int64_t>();
  const float *score_data = out_tensors[1].GetTensorData<float>();
  const float *desc_data = out_tensors[2].GetTensorData<float>();

  std::vector<int> selected;
  selected.reserve(static_cast<size_t>(n));
  const bool apply_conf_threshold = (min_confidence >= 0.0f);
  for (int i = 0; i < n; ++i) {
    if (!apply_conf_threshold || score_data[i] >= min_confidence) {
      selected.push_back(i);
    }
  }
  if (selected.empty()) {
    return;
  }
  const int n_selected = static_cast<int>(selected.size());
  const int capped_max_keypoints = std::max(1, max_keypoints);
  if (n_selected > capped_max_keypoints) {
    std::nth_element(selected.begin(), selected.begin() + capped_max_keypoints, selected.end(),
                     [&](int a, int b) { return score_data[a] > score_data[b]; });
    selected.resize(static_cast<size_t>(capped_max_keypoints));
  }

  const int out_n = static_cast<int>(selected.size());
  kpts.reserve(static_cast<size_t>(out_n));
  desc.create(out_n, d, CV_32F);
  float *desc_out = desc.ptr<float>();

  for (int i = 0; i < out_n; ++i) {
    const int idx = selected[static_cast<size_t>(i)];
    const float x = static_cast<float>(kpts_data[idx * 2 + 0]);
    const float y = static_cast<float>(kpts_data[idx * 2 + 1]);
    kpts.emplace_back(cv::Point2f(x, y), 1.0f);
    kpts.back().response = score_data[idx];
    std::memcpy(desc_out + static_cast<size_t>(i) * static_cast<size_t>(d), desc_data + static_cast<size_t>(idx) * static_cast<size_t>(d),
                sizeof(float) * static_cast<size_t>(d));
  }
}

void slg_backend::run_lightglue(const cv::Size &size0, const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, const cv::Size &size1,
                                const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1, std::vector<cv::DMatch> &matches,
                                float min_confidence, bool keypoints_normalized) const {
  matches.clear();
  if (kpts0.empty() || kpts1.empty() || desc0.empty() || desc1.empty()) {
    return;
  }
  if (desc0.rows != static_cast<int>(kpts0.size()) || desc1.rows != static_cast<int>(kpts1.size())) {
    throw std::runtime_error("slg_backend: descriptor/keypoint size mismatch");
  }
  if (desc0.cols != desc1.cols) {
    throw std::runtime_error("slg_backend: descriptor dimension mismatch");
  }
  if (desc0.type() != CV_32F || desc1.type() != CV_32F) {
    throw std::runtime_error("slg_backend: descriptors must be CV_32F");
  }

  const int64_t n0 = static_cast<int64_t>(kpts0.size());
  const int64_t n1 = static_cast<int64_t>(kpts1.size());
  const int64_t d = static_cast<int64_t>(desc0.cols);

  std::vector<cv::Point2f> n_kpts0;
  std::vector<cv::Point2f> n_kpts1;
  if (keypoints_normalized) {
    n_kpts0.reserve(kpts0.size());
    n_kpts1.reserve(kpts1.size());
    for (const cv::KeyPoint &kp : kpts0) {
      n_kpts0.emplace_back(kp.pt);
    }
    for (const cv::KeyPoint &kp : kpts1) {
      n_kpts1.emplace_back(kp.pt);
    }
  } else {
    const cv::Size resolved_size0 = infer_size_from_keypoints(size0, kpts0);
    const cv::Size resolved_size1 = infer_size_from_keypoints(size1, kpts1);
    n_kpts0 = normalize_keypoints(kpts0, resolved_size0.height, resolved_size0.width);
    n_kpts1 = normalize_keypoints(kpts1, resolved_size1.height, resolved_size1.width);
  }

  std::vector<float> kpts0_data(static_cast<size_t>(n0) * 2);
  std::vector<float> kpts1_data(static_cast<size_t>(n1) * 2);
  for (int64_t i = 0; i < n0; ++i) {
    kpts0_data[static_cast<size_t>(i) * 2 + 0] = n_kpts0[static_cast<size_t>(i)].x;
    kpts0_data[static_cast<size_t>(i) * 2 + 1] = n_kpts0[static_cast<size_t>(i)].y;
  }
  for (int64_t i = 0; i < n1; ++i) {
    kpts1_data[static_cast<size_t>(i) * 2 + 0] = n_kpts1[static_cast<size_t>(i)].x;
    kpts1_data[static_cast<size_t>(i) * 2 + 1] = n_kpts1[static_cast<size_t>(i)].y;
  }

  std::vector<int64_t> shape_kpts0{1, n0, 2};
  std::vector<int64_t> shape_kpts1{1, n1, 2};
  std::vector<int64_t> shape_desc0{1, n0, d};
  std::vector<int64_t> shape_desc1{1, n1, d};

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(
      Ort::Value::CreateTensor<float>(*impl_->memory_info, kpts0_data.data(), kpts0_data.size(), shape_kpts0.data(), shape_kpts0.size()));
  input_tensors.emplace_back(
      Ort::Value::CreateTensor<float>(*impl_->memory_info, kpts1_data.data(), kpts1_data.size(), shape_kpts1.data(), shape_kpts1.size()));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(*impl_->memory_info, const_cast<float *>(desc0.ptr<float>()), static_cast<size_t>(n0 * d),
                                                              shape_desc0.data(), shape_desc0.size()));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(*impl_->memory_info, const_cast<float *>(desc1.ptr<float>()), static_cast<size_t>(n1 * d),
                                                              shape_desc1.data(), shape_desc1.size()));

  const std::vector<const char *> in_names = to_cstr_ptrs(impl_->lightglue_input_names);
  const std::vector<const char *> out_names = to_cstr_ptrs(impl_->lightglue_output_names);

  std::vector<Ort::Value> out_tensors = impl_->lightglue_session->Run(Ort::RunOptions{nullptr}, in_names.data(), input_tensors.data(),
                                                                       input_tensors.size(), out_names.data(), out_names.size());
  if (out_tensors.size() < 2) {
    throw std::runtime_error("slg_backend: LightGlue output tensor count is less than 2");
  }

  const auto matches_shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto score_shape = out_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
  if (matches_shape.size() != 2 || matches_shape[1] != 2 || score_shape.empty()) {
    throw std::runtime_error("slg_backend: unexpected LightGlue output shape");
  }

  const int64_t m = matches_shape[0];
  const int64_t *match_data = out_tensors[0].GetTensorData<int64_t>();
  const float *score_data = out_tensors[1].GetTensorData<float>();

  matches.reserve(static_cast<size_t>(m));
  const bool apply_conf_threshold = (min_confidence >= 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    const int q = static_cast<int>(match_data[i * 2 + 0]);
    const int t = static_cast<int>(match_data[i * 2 + 1]);
    if (q < 0 || t < 0 || q >= static_cast<int>(n0) || t >= static_cast<int>(n1)) {
      continue;
    }
    const float score = score_data[i];
    const float clamped = std::max(0.0f, std::min(1.0f, score));
    if (apply_conf_threshold && clamped < min_confidence) {
      continue;
    }
    matches.emplace_back(q, t, 1.0f - clamped);
  }
}

} // namespace ov_lightglue
