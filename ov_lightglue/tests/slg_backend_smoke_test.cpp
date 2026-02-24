/*
 * Standalone smoke test for ov_lightglue slg_backend.
 */

#include "track/slg_backend.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <superpoint.onnx> <lightglue.onnx> <image0> <image1> [use_gpu=1]" << std::endl;
    return 2;
  }

  const std::string superpoint_onnx_path = argv[1];
  const std::string lightglue_onnx_path = argv[2];
  const bool use_gpu = (argc >= 6) ? (std::atoi(argv[5]) != 0) : true;

  const cv::Mat img0 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
  const cv::Mat img1 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
  if (img0.empty() || img1.empty()) {
    std::cerr << "Failed to load input images." << std::endl;
    return 3;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    ov_lightglue::slg_backend backend(superpoint_onnx_path, lightglue_onnx_path, use_gpu, ov_lightglue::slg_backend::log_level::info);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<cv::KeyPoint> kpts0, kpts1;
    cv::Mat desc0, desc1;
    backend.run_superpoint(img0, kpts0, desc0, 1024, -1.0f);
    backend.run_superpoint(img1, kpts1, desc1, 1024, -1.0f);
    const auto t2 = std::chrono::steady_clock::now();

    std::vector<cv::DMatch> matches;
    backend.run_lightglue(img0.size(), kpts0, desc0, img1.size(), kpts1, desc1, matches, -1.0f);
    const auto t3 = std::chrono::steady_clock::now();

    const double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double sp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double lg_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "slg_backend smoke test success\n";
    std::cout << "init_ms=" << init_ms << " superpoint_ms=" << sp_ms << " lightglue_ms=" << lg_ms << "\n";
    std::cout << "kpts0=" << kpts0.size() << " kpts1=" << kpts1.size() << " desc0=" << desc0.rows << "x" << desc0.cols << " desc1=" << desc1.rows
              << "x" << desc1.cols << " matches=" << matches.size() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "slg_backend smoke test failed: " << e.what() << std::endl;
    return 4;
  }

  return 0;
}
