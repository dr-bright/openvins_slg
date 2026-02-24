/*
 * Standalone ONNX Runtime execution provider check.
 */

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool has_provider(const std::vector<std::string> &providers, const std::string &name) {
  return std::find(providers.begin(), providers.end(), name) != providers.end();
}

OrtLoggingLevel parse_verbosity(int argc, char **argv) {
  if (argc < 2) {
    return ORT_LOGGING_LEVEL_WARNING;
  }
  const std::string arg(argv[1]);
  if (arg == "warning") {
    return ORT_LOGGING_LEVEL_WARNING;
  }
  if (arg == "info") {
    return ORT_LOGGING_LEVEL_INFO;
  }
  if (arg == "verbose") {
    return ORT_LOGGING_LEVEL_VERBOSE;
  }
  return ORT_LOGGING_LEVEL_WARNING;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const OrtLoggingLevel verbosity = parse_verbosity(argc, argv);
    Ort::Env env(verbosity, "ov_lightglue_onnxrt_check");
    (void)env;

    Ort::SessionOptions session_options;
    session_options.SetLogSeverityLevel(static_cast<int>(verbosity));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    std::cout << "Providers:";
    for (const std::string &provider : providers) {
      std::cout << " " << provider;
    }
    std::cout << std::endl;

    const bool cuda_available = has_provider(providers, "CUDAExecutionProvider");
    if (!cuda_available) {
      std::cout << "Reason: CUDAExecutionProvider is not in available providers list." << std::endl;
    }

    bool cuda_append_ok = false;
    try {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      cuda_append_ok = true;
      std::cout << "CUDA append test: success" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "CUDA append test: failed - " << e.what() << std::endl;
    }

    if (cuda_available && cuda_append_ok) {
      std::cout << "CUDA" << std::endl;
    } else {
      std::cout << "CPU" << std::endl;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "onnxrt_check failed: " << e.what() << std::endl;
    return 1;
  }
}
