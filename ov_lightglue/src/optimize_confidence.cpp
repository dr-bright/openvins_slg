/*
 * Confidence threshold optimizer for TrackSuperLightGlue.
 *
 * This tool uses simulated annealing (SA) to tune:
 *   - detect_min_confidence (d)
 *   - match_min_confidence  (m)
 *
 * Objective: maximize a balanced quality metric that combines
 * feature count and temporal propagation quality.
 *
 * Optional outputs are enabled by passing non-empty CLI paths:
 *   - iteration CSV log (--iter-csv)
 *   - per-frame CSV log for final/best replay (--frame-csv)
 *   - MP4 visualization for final/best replay (--mp4)
 */

#include "cam/CamRadtan.h"
#include "track/TrackSuperLightGlue.h"
#include "track/TrackerSuperLightGlueConfig.h"
#include "utils/sensor_data.h"

#include <Eigen/Core>

#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct Frame {
  double timestamp = 0.0;
  cv::Mat gray;
};

struct SAParams {
  float d = -1.0f;
  float m = -1.0f;
};

struct EvalSummary {
  double mean_active = 0.0;
  double mean_prop = 0.0;
  double quality = -std::numeric_limits<double>::infinity();
  size_t frames = 0;
};

struct Settings {
  std::string superpoint_onnx_path;
  std::string lightglue_onnx_path;
  bool use_gpu = false;

  size_t window_size = 60;
  size_t max_iterations = 40;
  double temperature_init = 0.2;
  double temperature_min = 0.01;
  double temperature_alpha = 0.95;
  double sigma_d = 0.15;
  double sigma_m = 0.15;
  unsigned int seed = 1;

  float d_min = -1.0f;
  float d_max = 0.99f;
  float m_min = -1.0f;
  float m_max = 0.99f;

  int target_active_features = 300;

  size_t max_frames = 0;

  std::string iter_csv_path;
  std::string frame_csv_path;
  std::string mp4_path;
};

class TrackSuperLightGlueExposed final : public ov_lightglue::TrackSuperLightGlue {
public:
  using ov_lightglue::TrackSuperLightGlue::TrackSuperLightGlue;

  void set_confidences(float detect_min_confidence, float match_min_confidence) {
    config_.detect_min_confidence = detect_min_confidence;
    config_.match_min_confidence = match_min_confidence;
  }
};

cv::Mat decode_grayscale(const sensor_msgs::ImageConstPtr &msg) {
  if (!msg) {
    return cv::Mat();
  }

  cv::Mat gray;
  try {
    if (msg->encoding == sensor_msgs::image_encodings::MONO8 || msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
      gray = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8)->image;
    } else if (msg->encoding == sensor_msgs::image_encodings::MONO16 || msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv::Mat mono16 = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16)->image;
      mono16.convertTo(gray, CV_8U, 1.0 / 256.0);
    } else {
      cv::Mat bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    }
  } catch (...) {
    return cv::Mat();
  }

  return gray;
}

std::vector<Frame> load_frames(const std::string &bag_path, const std::string &topic, size_t max_frames) {
  std::vector<Frame> frames;

  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View view(bag, rosbag::TopicQuery(topic));

  for (const rosbag::MessageInstance &m : view) {
    sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
    if (!img_msg) {
      continue;
    }

    cv::Mat gray = decode_grayscale(img_msg);
    if (gray.empty()) {
      continue;
    }

    Frame f;
    f.timestamp = img_msg->header.stamp.toSec();
    f.gray = gray;
    frames.push_back(f);

    if (max_frames > 0 && frames.size() >= max_frames) {
      break;
    }
  }

  bag.close();
  return frames;
}

double infer_fps(const std::vector<Frame> &frames) {
  if (frames.size() < 2) {
    return 0.0;
  }

  std::vector<double> dts;
  dts.reserve(frames.size() - 1);
  for (size_t i = 1; i < frames.size(); ++i) {
    const double dt = frames[i].timestamp - frames[i - 1].timestamp;
    if (dt > 1e-6) {
      dts.push_back(dt);
    }
  }
  if (dts.empty()) {
    return 0.0;
  }

  std::sort(dts.begin(), dts.end());
  const double median_dt = dts[dts.size() / 2];
  return (median_dt > 1e-6) ? (1.0 / median_dt) : 0.0;
}

float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}

std::unique_ptr<TrackSuperLightGlueExposed> create_tracker(const cv::Size &size, const Settings &s, const SAParams &p) {
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
  auto cam = std::make_shared<ov_core::CamRadtan>(size.width, size.height);
  Eigen::Matrix<double, 8, 1> calib;
  calib << 300.0, 300.0, static_cast<double>(size.width) * 0.5, static_cast<double>(size.height) * 0.5, 0.0, 0.0, 0.0, 0.0;
  cam->set_value(calib);
  cameras.insert({0, cam});

  ov_lightglue::TrackSuperLightGlueConfig cfg;
  cfg.superpoint_onnx_path = s.superpoint_onnx_path;
  cfg.lightglue_onnx_path = s.lightglue_onnx_path;
  cfg.use_gpu = s.use_gpu;
  cfg.max_keypoints = s.target_active_features;
  cfg.detect_min_confidence = p.d;
  cfg.match_min_confidence = p.m;

  auto tracker = std::unique_ptr<TrackSuperLightGlueExposed>(
      new TrackSuperLightGlueExposed(cameras, s.target_active_features, 0, false, ov_core::TrackBase::HistogramMethod::NONE, cfg));
  tracker->set_confidences(p.d, p.m);
  return tracker;
}

/*
 * Quality metric math (window-level)
 * ----------------------------------
 * We optimize two objectives:
 *   A = mean active feature count
 *   P = mean propagation ratio = mean(carried_t / active_t)
 *
 * A and P live on different scales, so we normalize active count:
 *   A_n = clamp(A / A_target, 0, 1)
 *
 * P already lies in [0,1], so:
 *   P_n = clamp(P, 0, 1)
 *
 * The final scalar objective is the harmonic mean:
 *
 *             2 A_n P_n
 *   q = ------------------------
 *        A_n + P_n + epsilon
 *
 * Why harmonic mean?
 * - It rewards balance.
 * - It strongly penalizes one objective collapsing to near-zero.
 * - It prevents a trivial high-count / low-propagation solution from dominating.
 */
double compute_quality(double mean_active, double mean_prop, int target_active) {
  const double eps = 1e-9;
  const double A_n = std::max(0.0, std::min(1.0, mean_active / std::max(1, target_active)));
  const double P_n = std::max(0.0, std::min(1.0, mean_prop));
  return (2.0 * A_n * P_n) / (A_n + P_n + eps);
}

EvalSummary evaluate_window(const std::vector<Frame> &frames, size_t start, size_t window, const Settings &s, const SAParams &p) {
  EvalSummary out;
  if (frames.empty() || start >= frames.size()) {
    return out;
  }

  const size_t end = std::min(frames.size(), start + window);
  if (end <= start) {
    return out;
  }

  auto tracker = create_tracker(frames[start].gray.size(), s, p);

  std::unordered_map<size_t, cv::Point2f> prev_pts_by_id;
  double sum_active = 0.0;
  double sum_prop = 0.0;

  for (size_t i = start; i < end; ++i) {
    ov_core::CameraData data;
    data.timestamp = frames[i].timestamp;
    data.sensor_ids.push_back(0);
    data.images.push_back(frames[i].gray);
    data.masks.push_back(cv::Mat::zeros(frames[i].gray.rows, frames[i].gray.cols, CV_8UC1));
    tracker->feed_new_camera(data);

    const auto ids_by_cam = tracker->get_last_ids();
    const auto obs_by_cam = tracker->get_last_obs();
    const std::vector<size_t> curr_ids = (ids_by_cam.count(0) > 0) ? ids_by_cam.at(0) : std::vector<size_t>{};
    const std::vector<cv::KeyPoint> curr_kpts = (obs_by_cam.count(0) > 0) ? obs_by_cam.at(0) : std::vector<cv::KeyPoint>{};

    size_t carried = 0;
    for (size_t j = 0; j < curr_ids.size() && j < curr_kpts.size(); ++j) {
      if (prev_pts_by_id.find(curr_ids[j]) != prev_pts_by_id.end()) {
        ++carried;
      }
    }

    const double active = static_cast<double>(curr_ids.size());
    const double prop = (active > 1e-9) ? (static_cast<double>(carried) / active) : 0.0;

    sum_active += active;
    sum_prop += prop;
    out.frames++;

    prev_pts_by_id.clear();
    prev_pts_by_id.reserve(curr_ids.size());
    for (size_t j = 0; j < curr_ids.size() && j < curr_kpts.size(); ++j) {
      prev_pts_by_id[curr_ids[j]] = curr_kpts[j].pt;
    }
  }

  if (out.frames == 0) {
    return out;
  }

  out.mean_active = sum_active / static_cast<double>(out.frames);
  out.mean_prop = sum_prop / static_cast<double>(out.frames);
  out.quality = compute_quality(out.mean_active, out.mean_prop, s.target_active_features);
  return out;
}

void draw_overlay(cv::Mat &frame_bgr, const std::vector<size_t> &curr_ids, const std::vector<cv::KeyPoint> &curr_kpts,
                  const std::unordered_map<size_t, cv::Point2f> &prev_pts_by_id, size_t frame_index, const SAParams &best,
                  double quality) {
  size_t carried = 0;

  for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
    const size_t id = curr_ids[i];
    const cv::Point2f curr = curr_kpts[i].pt;

    auto it = prev_pts_by_id.find(id);
    if (it != prev_pts_by_id.end()) {
      ++carried;
      cv::line(frame_bgr, it->second, curr, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
    } else {
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
  }

  const size_t active = curr_ids.size();
  const size_t fresh = (active >= carried) ? (active - carried) : 0;

  const std::string line1 = "frame=" + std::to_string(frame_index) + " active=" + std::to_string(active) + " carried=" +
                            std::to_string(carried) + " new=" + std::to_string(fresh);
  const std::string line2 = "best_d=" + cv::format("%.4f", best.d) + " best_m=" + cv::format("%.4f", best.m) +
                            " best_q=" + cv::format("%.4f", quality);

  cv::putText(frame_bgr, line1, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::putText(frame_bgr, line2, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void replay_best_and_dump(const std::vector<Frame> &frames, const Settings &s, const SAParams &best, double best_quality) {
  if (frames.empty()) {
    return;
  }

  std::ofstream frame_csv;
  if (!s.frame_csv_path.empty()) {
    frame_csv.open(s.frame_csv_path);
    if (!frame_csv.is_open()) {
      throw std::runtime_error("Failed to open frame csv: " + s.frame_csv_path);
    }
    frame_csv << "frame,timestamp,active,carried,propagation,best_d,best_m,best_q\n";
  }

  cv::VideoWriter writer;
  if (!s.mp4_path.empty()) {
    const double fps = infer_fps(frames);
    if (fps <= 0.0) {
      throw std::runtime_error("Unable to infer FPS for mp4 output.");
    }
    writer.open(s.mp4_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frames.front().gray.size(), true);
    if (!writer.isOpened()) {
      throw std::runtime_error("Failed to open mp4 writer: " + s.mp4_path);
    }
  }

  auto tracker = create_tracker(frames.front().gray.size(), s, best);
  std::unordered_map<size_t, cv::Point2f> prev_pts_by_id;

  for (size_t i = 0; i < frames.size(); ++i) {
    ov_core::CameraData data;
    data.timestamp = frames[i].timestamp;
    data.sensor_ids.push_back(0);
    data.images.push_back(frames[i].gray);
    data.masks.push_back(cv::Mat::zeros(frames[i].gray.rows, frames[i].gray.cols, CV_8UC1));
    tracker->feed_new_camera(data);

    const auto ids_by_cam = tracker->get_last_ids();
    const auto obs_by_cam = tracker->get_last_obs();
    const std::vector<size_t> curr_ids = (ids_by_cam.count(0) > 0) ? ids_by_cam.at(0) : std::vector<size_t>{};
    const std::vector<cv::KeyPoint> curr_kpts = (obs_by_cam.count(0) > 0) ? obs_by_cam.at(0) : std::vector<cv::KeyPoint>{};

    size_t carried = 0;
    for (size_t j = 0; j < curr_ids.size() && j < curr_kpts.size(); ++j) {
      if (prev_pts_by_id.find(curr_ids[j]) != prev_pts_by_id.end()) {
        ++carried;
      }
    }

    const size_t active = curr_ids.size();
    const double prop = (active > 0) ? (static_cast<double>(carried) / static_cast<double>(active)) : 0.0;

    if (frame_csv.is_open()) {
      frame_csv << i << "," << frames[i].timestamp << "," << active << "," << carried << "," << prop << "," << best.d << "," << best.m
                << "," << best_quality << "\n";
    }

    if (writer.isOpened()) {
      cv::Mat frame_bgr;
      cv::cvtColor(frames[i].gray, frame_bgr, cv::COLOR_GRAY2BGR);
      draw_overlay(frame_bgr, curr_ids, curr_kpts, prev_pts_by_id, i, best, best_quality);
      writer.write(frame_bgr);
    }

    prev_pts_by_id.clear();
    prev_pts_by_id.reserve(curr_ids.size());
    for (size_t j = 0; j < curr_ids.size() && j < curr_kpts.size(); ++j) {
      prev_pts_by_id[curr_ids[j]] = curr_kpts[j].pt;
    }
  }

  if (frame_csv.is_open()) {
    frame_csv.close();
  }
  if (writer.isOpened()) {
    writer.release();
  }
}

bool parse_args(int argc, char **argv, Settings &s, std::string &bag_path, std::string &topic, float &d_init, float &m_init) {
  if (argc < 6) {
    return false;
  }

  s.superpoint_onnx_path = argv[1];
  s.lightglue_onnx_path = argv[2];
  s.use_gpu = (std::atoi(argv[3]) != 0);
  bag_path = argv[4];
  topic = argv[5];

  for (int i = 6; i < argc; ++i) {
    const std::string arg = argv[i];

    auto need_value = [&](const std::string &name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--window") {
      s.window_size = static_cast<size_t>(std::stoul(need_value(arg)));
    } else if (arg == "--iters") {
      s.max_iterations = static_cast<size_t>(std::stoul(need_value(arg)));
    } else if (arg == "--d-init") {
      d_init = static_cast<float>(std::stof(need_value(arg)));
    } else if (arg == "--m-init") {
      m_init = static_cast<float>(std::stof(need_value(arg)));
    } else if (arg == "--t-init") {
      s.temperature_init = std::stod(need_value(arg));
    } else if (arg == "--t-min") {
      s.temperature_min = std::stod(need_value(arg));
    } else if (arg == "--alpha") {
      s.temperature_alpha = std::stod(need_value(arg));
    } else if (arg == "--sigma-d") {
      s.sigma_d = std::stod(need_value(arg));
    } else if (arg == "--sigma-m") {
      s.sigma_m = std::stod(need_value(arg));
    } else if (arg == "--seed") {
      s.seed = static_cast<unsigned int>(std::stoul(need_value(arg)));
    } else if (arg == "--target-active") {
      s.target_active_features = std::max(1, std::stoi(need_value(arg)));
    } else if (arg == "--max-frames") {
      s.max_frames = static_cast<size_t>(std::stoul(need_value(arg)));
    } else if (arg == "--iter-csv") {
      s.iter_csv_path = need_value(arg);
    } else if (arg == "--frame-csv") {
      s.frame_csv_path = need_value(arg);
    } else if (arg == "--mp4") {
      s.mp4_path = need_value(arg);
    } else if (arg == "--d-min") {
      s.d_min = static_cast<float>(std::stof(need_value(arg)));
    } else if (arg == "--d-max") {
      s.d_max = static_cast<float>(std::stof(need_value(arg)));
    } else if (arg == "--m-min") {
      s.m_min = static_cast<float>(std::stof(need_value(arg)));
    } else if (arg == "--m-max") {
      s.m_max = static_cast<float>(std::stof(need_value(arg)));
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return true;
}

void print_usage(const char *prog) {
  std::cout
      << "Usage:\n  " << prog
      << " <superpoint.onnx> <lightglue.onnx> <use_gpu={0,1}> <bag> <image_topic> [options]\n\n"
      << "Options:\n"
      << "  --window N          Window size in frames (default: 60)\n"
      << "  --iters N           SA iterations (default: 40)\n"
      << "  --d-init X          Initial detect_min_confidence (default: -1.0)\n"
      << "  --m-init X          Initial match_min_confidence  (default: -1.0)\n"
      << "  --t-init X          Initial temperature T0 (default: 0.2)\n"
      << "  --t-min X           Minimum temperature (default: 0.01)\n"
      << "  --alpha X           Temperature decay factor (default: 0.95)\n"
      << "  --sigma-d X         Proposal sigma for detect threshold (default: 0.15)\n"
      << "  --sigma-m X         Proposal sigma for match threshold  (default: 0.15)\n"
      << "  --seed N            RNG seed (default: 1)\n"
      << "  --target-active N   Target active features for normalization (default: 300)\n"
      << "  --max-frames N      Limit decoded frames from bag (default: all)\n"
      << "  --d-min X           Lower clamp bound for d (default: -1.0)\n"
      << "  --d-max X           Upper clamp bound for d (default: 0.99)\n"
      << "  --m-min X           Lower clamp bound for m (default: -1.0)\n"
      << "  --m-max X           Upper clamp bound for m (default: 0.99)\n"
      << "  --iter-csv PATH     Optional SA iteration log CSV\n"
      << "  --frame-csv PATH    Optional per-frame CSV for final/best replay\n"
      << "  --mp4 PATH          Optional MP4 visualization for final/best replay\n";
}

} // namespace

int main(int argc, char **argv) {
  Settings s;
  std::string bag_path;
  std::string topic;

  float d_init = -1.0f;
  float m_init = -1.0f;

  try {
    if (!parse_args(argc, argv, s, bag_path, topic, d_init, m_init)) {
      print_usage(argv[0]);
      return 2;
    }

    d_init = clampf(d_init, s.d_min, s.d_max);
    m_init = clampf(m_init, s.m_min, s.m_max);

    if (s.window_size == 0) {
      throw std::runtime_error("window must be > 0");
    }
    if (s.max_iterations == 0) {
      throw std::runtime_error("iters must be > 0");
    }
    if (s.temperature_init <= 0.0 || s.temperature_min <= 0.0 || s.temperature_alpha <= 0.0 || s.temperature_alpha > 1.0) {
      throw std::runtime_error("Invalid annealing parameters");
    }

    std::cout << "Loading frames from bag..." << std::endl;
    const std::vector<Frame> frames = load_frames(bag_path, topic, s.max_frames);
    if (frames.size() < 2) {
      throw std::runtime_error("Not enough decodable image frames.");
    }
    std::cout << "Loaded frames=" << frames.size() << std::endl;

    std::ofstream iter_csv;
    if (!s.iter_csv_path.empty()) {
      iter_csv.open(s.iter_csv_path);
      if (!iter_csv.is_open()) {
        throw std::runtime_error("Failed to open iteration csv: " + s.iter_csv_path);
      }
      iter_csv << "iter,start,window,T,d,m,q,mean_active,mean_prop,d_prop,m_prop,q_prop,mean_active_prop,mean_prop_prop,accepted,delta\n";
    }

    std::mt19937 rng(s.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    SAParams current{d_init, m_init};
    SAParams best = current;
    double best_q = -std::numeric_limits<double>::infinity();
    double T = s.temperature_init;

    const size_t max_start = (frames.size() > s.window_size) ? (frames.size() - s.window_size) : 0;
    std::uniform_int_distribution<size_t> start_dist(0, max_start);

    for (size_t iter = 0; iter < s.max_iterations; ++iter) {
      const size_t start = (max_start > 0) ? start_dist(rng) : 0;

      const EvalSummary eval_curr = evaluate_window(frames, start, s.window_size, s, current);
      if (eval_curr.frames == 0) {
        throw std::runtime_error("Current evaluation returned 0 frames.");
      }

      const double scale = std::sqrt(std::max(T, 1e-12) / s.temperature_init);
      std::normal_distribution<double> nd(0.0, s.sigma_d * scale);
      std::normal_distribution<double> nm(0.0, s.sigma_m * scale);

      SAParams proposal;
      proposal.d = clampf(static_cast<float>(current.d + nd(rng)), s.d_min, s.d_max);
      proposal.m = clampf(static_cast<float>(current.m + nm(rng)), s.m_min, s.m_max);

      const EvalSummary eval_prop = evaluate_window(frames, start, s.window_size, s, proposal);
      if (eval_prop.frames == 0) {
        throw std::runtime_error("Proposal evaluation returned 0 frames.");
      }

      const double delta = eval_prop.quality - eval_curr.quality;

      // Simulated annealing acceptance rule:
      // 1) Always accept better candidate (delta >= 0)
      // 2) Accept worse candidate with probability exp(delta / T), delta < 0.
      //    This permits occasional uphill moves in the "energy" landscape,
      //    helping escape local optima early in the search.
      bool accept = false;
      if (delta >= 0.0) {
        accept = true;
      } else {
        const double p_accept = std::exp(delta / std::max(T, 1e-12));
        accept = (unif01(rng) < p_accept);
      }

      if (accept) {
        current = proposal;
      }

      const SAParams &effective = accept ? proposal : current;
      const EvalSummary &eval_effective = accept ? eval_prop : eval_curr;
      if (eval_effective.quality > best_q) {
        best_q = eval_effective.quality;
        best = effective;
      }

      if (iter_csv.is_open()) {
        iter_csv << iter << "," << start << "," << s.window_size << "," << T << "," << current.d << "," << current.m << ","
                 << eval_curr.quality << "," << eval_curr.mean_active << "," << eval_curr.mean_prop << "," << proposal.d << ","
                 << proposal.m << ","
                 << eval_prop.quality << "," << eval_prop.mean_active << "," << eval_prop.mean_prop << "," << (accept ? 1 : 0) << ","
                 << delta << "\n";
      }

      std::cout << "iter=" << iter << " T=" << T << " curr(d,m)=(" << current.d << ", " << current.m << ")"
                << " q_curr=" << eval_curr.quality << " q_prop=" << eval_prop.quality << " accept=" << (accept ? "yes" : "no")
                << " best_q=" << best_q << std::endl;

      T = std::max(s.temperature_min, T * s.temperature_alpha);
    }

    if (iter_csv.is_open()) {
      iter_csv.close();
    }

    std::cout << "\nBest thresholds:\n"
              << "  detect_min_confidence = " << best.d << "\n"
              << "  match_min_confidence  = " << best.m << "\n"
              << "  quality               = " << best_q << std::endl;

    if (!s.frame_csv_path.empty() || !s.mp4_path.empty()) {
      std::cout << "Replaying with best parameters for optional outputs..." << std::endl;
      replay_best_and_dump(frames, s, best, best_q);
      if (!s.frame_csv_path.empty()) {
        std::cout << "Wrote frame CSV: " << s.frame_csv_path << std::endl;
      }
      if (!s.mp4_path.empty()) {
        std::cout << "Wrote mp4: " << s.mp4_path << std::endl;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "optimize_confidence failed: " << e.what() << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}
