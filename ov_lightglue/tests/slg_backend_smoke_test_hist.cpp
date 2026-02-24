/*
 * Standalone histogram smoke test for ov_lightglue slg_backend.
 */

#include "track/slg_backend.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace {

struct HistStats {
  float min_v = 0.0f;
  float max_v = 0.0f;
  float mean_v = 0.0f;
};

HistStats compute_stats(const std::vector<float> &values) {
  HistStats stats;
  if (values.empty()) {
    return stats;
  }
  stats.min_v = values.front();
  stats.max_v = values.front();
  double sum = 0.0;
  for (float v : values) {
    stats.min_v = std::min(stats.min_v, v);
    stats.max_v = std::max(stats.max_v, v);
    sum += static_cast<double>(v);
  }
  stats.mean_v = static_cast<float>(sum / static_cast<double>(values.size()));
  return stats;
}

std::vector<int> histogram20(const std::vector<float> &values) {
  std::vector<int> bins(20, 0);
  for (float v : values) {
    int idx = static_cast<int>(v * 20.0f);
    if (idx < 0) {
      idx = 0;
    }
    if (idx > 19) {
      idx = 19;
    }
    bins[static_cast<size_t>(idx)]++;
  }
  return bins;
}

void print_hist(const std::string &name, const std::vector<float> &values) {
  const HistStats stats = compute_stats(values);
  const std::vector<int> bins = histogram20(values);

  std::cout << name << "_count=" << values.size() << " min=" << std::fixed << std::setprecision(4) << stats.min_v << " max=" << stats.max_v
            << " mean=" << stats.mean_v << "\n";
  std::cout << name << "_hist20:";
  for (size_t i = 0; i < bins.size(); ++i) {
    std::cout << (i == 0 ? " " : ",") << bins[i];
  }
  std::cout << "\n";
}

bool write_csv(const std::string &path, const std::vector<float> &sp0, const std::vector<float> &sp1, const std::vector<float> &m) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }
  out << "series,value\n";
  for (float v : sp0) {
    out << "sp0," << v << "\n";
  }
  for (float v : sp1) {
    out << "sp1," << v << "\n";
  }
  for (float v : m) {
    out << "match," << v << "\n";
  }
  return true;
}

void maybe_plot_histograms(const std::vector<float> &sp0, const std::vector<float> &sp1, const std::vector<float> &m) {
  const std::string csv_path = "/tmp/slg_backend_smoke_test_hist.csv";
  if (!write_csv(csv_path, sp0, sp1, m)) {
    std::cerr << "Failed to write CSV for plotting: " << csv_path << std::endl;
    return;
  }

  const std::string script =
      "import csv,sys\n"
      "import matplotlib.pyplot as plt\n"
      "p=sys.argv[1]\n"
      "s={'sp0':[],'sp1':[],'match':[]}\n"
      "with open(p,'r') as f:\n"
      "  r=csv.DictReader(f)\n"
      "  for row in r:\n"
      "    s[row['series']].append(float(row['value']))\n"
      "fig,ax=plt.subplots(1,3,figsize=(14,4))\n"
      "ax[0].hist(s['sp0'],bins=20,range=(0,1),color='tab:red')\n"
      "ax[0].set_title('Frame 0')\n"
      "ax[1].hist(s['sp1'],bins=20,range=(0,1),color='tab:blue')\n"
      "ax[1].set_title('Frame 1')\n"
      "ax[2].hist(s['match'],bins=20,range=(0,1),color='tab:green')\n"
      "ax[2].set_title('Matches')\n"
      "for a in ax:\n"
      "  a.set_xlim(0,1)\n"
      "  a.set_xlabel('confidence')\n"
      "plt.tight_layout()\n"
      "plt.show()\n";

  const std::string cmd =
      "python3 -c \"" + script + "\" " + csv_path +
      " >/tmp/slg_backend_smoke_test_hist_plot.log 2>&1";
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::cerr << "Plotting failed (python3/matplotlib missing or no display). See /tmp/slg_backend_smoke_test_hist_plot.log" << std::endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <superpoint.onnx> <lightglue.onnx> <image0> <image1> [use_gpu=1] [--plot]" << std::endl;
    return 2;
  }

  const std::string superpoint_onnx_path = argv[1];
  const std::string lightglue_onnx_path = argv[2];
  const bool use_gpu = (argc >= 6) ? (std::atoi(argv[5]) != 0) : true;

  bool plot = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--plot") {
      plot = true;
    }
  }

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

    std::vector<float> sp_conf0;
    std::vector<float> sp_conf1;
    sp_conf0.reserve(kpts0.size());
    sp_conf1.reserve(kpts1.size());
    for (const cv::KeyPoint &kp : kpts0) {
      sp_conf0.push_back(kp.response);
    }
    for (const cv::KeyPoint &kp : kpts1) {
      sp_conf1.push_back(kp.response);
    }

    std::vector<float> match_conf;
    match_conf.reserve(matches.size());
    for (const cv::DMatch &m : matches) {
      match_conf.push_back(1.0f - m.distance);
    }

    const double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double sp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double lg_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "slg_backend histogram smoke test success\n";
    std::cout << "init_ms=" << init_ms << " superpoint_ms=" << sp_ms << " lightglue_ms=" << lg_ms << "\n";
    std::cout << "kpts0=" << kpts0.size() << " kpts1=" << kpts1.size() << " desc0=" << desc0.rows << "x" << desc0.cols << " desc1="
              << desc1.rows << "x" << desc1.cols << " matches=" << matches.size() << std::endl;

    print_hist("frame0_conf", sp_conf0);
    print_hist("frame1_conf", sp_conf1);
    print_hist("match_conf", match_conf);

    if (plot) {
      maybe_plot_histograms(sp_conf0, sp_conf1, match_conf);
    }
  } catch (const std::exception &e) {
    std::cerr << "slg_backend histogram smoke test failed: " << e.what() << std::endl;
    return 4;
  }

  return 0;
}
