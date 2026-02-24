/*
 * Standalone smoke test for ov_lightglue slg_backend.
 */

#include "track/TrackSuperLightGlue.h"
#include "track/slg_backend.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <superpoint.onnx> <lightglue.onnx> <image0> <image1> [use_gpu=1]" << std::endl;
    return 2;
  }

  ov_lightglue::slg_config cfg;
  cfg.superpoint_onnx_path = argv[1];
  cfg.lightglue_onnx_path = argv[2];
  cfg.use_gpu = (argc >= 6) ? (std::atoi(argv[5]) != 0) : true;

  const cv::Mat img0 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
  const cv::Mat img1 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
  if (img0.empty() || img1.empty()) {
    std::cerr << "Failed to load input images." << std::endl;
    return 3;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    ov_lightglue::slg_backend backend(cfg, ORT_LOGGING_LEVEL_INFO);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<cv::KeyPoint> kpts0, kpts1;
    cv::Mat desc0, desc1;
    backend.run_superpoint(img0, kpts0, desc0);
    backend.run_superpoint(img1, kpts1, desc1);
    const auto t2 = std::chrono::steady_clock::now();

    std::vector<cv::DMatch> matches;
    backend.run_lightglue(kpts0, desc0, kpts1, desc1, matches);
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
