/*
 * Rosbag to MP4 visualization tool for TrackSuperLightGlue.
 */

#include "cam/CamRadtan.h"
#include "track/TrackSuperLightGlue.h"
#include "utils/sensor_data.h"

#include <Eigen/Core>

#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

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

void draw_overlay(cv::Mat &frame_bgr, const std::vector<size_t> &curr_ids, const std::vector<cv::KeyPoint> &curr_kpts,
                  const std::unordered_map<size_t, cv::Point2f> &prev_pts_by_id, size_t frame_index) {
  size_t carried = 0;

  for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
    const size_t id = curr_ids[i];
    const cv::Point2f curr = curr_kpts[i].pt;

    auto it = prev_pts_by_id.find(id);
    if (it != prev_pts_by_id.end()) {
      ++carried;
      cv::line(frame_bgr, it->second, curr, cv::Scalar(255, 0, 0), 1, cv::LINE_AA); // blue match line
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA); // green carried
    } else {
      cv::circle(frame_bgr, curr, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA); // red new/current
    }
  }

  const size_t active = curr_ids.size();
  const size_t fresh = (active >= carried) ? (active - carried) : 0;
  const std::string text = "frame=" + std::to_string(frame_index) + " active=" + std::to_string(active) + " carried=" +
                           std::to_string(carried) + " new=" + std::to_string(fresh);
  cv::putText(frame_bgr, text, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " <superpoint.onnx> <lightglue.onnx> <use_gpu={0,1}> <bag> <image_topic> <output.mp4> <output.csv> [fps=20]" << std::endl;
    return 2;
  }

  const std::string superpoint_onnx_path = argv[1];
  const std::string lightglue_onnx_path = argv[2];
  const bool use_gpu = (std::atoi(argv[3]) != 0);
  const std::string bag_path = argv[4];
  const std::string topic = argv[5];
  const std::string output_mp4 = argv[6];
  const std::string output_csv = argv[7];
  const double out_fps = (argc >= 9) ? std::atof(argv[8]) : 20.0;

  if (out_fps <= 0.0) {
    std::cerr << "Invalid fps value: " << out_fps << std::endl;
    return 2;
  }

  try {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topic));

    if (view.size() == 0) {
      std::cerr << "No messages found for topic: " << topic << std::endl;
      return 3;
    }

    bool tracker_initialized = false;
    bool writer_initialized = false;

    std::unique_ptr<ov_lightglue::TrackSuperLightGlue> tracker;
    std::unordered_map<size_t, cv::Point2f> prev_pts_by_id;
    cv::VideoWriter writer;
    std::ofstream metrics(output_csv);
    if (!metrics.is_open()) {
      std::cerr << "Failed to open metrics csv: " << output_csv << std::endl;
      return 4;
    }
    metrics << "frame,timestamp,active,carried,new\n";

    size_t frame_idx = 0;
    size_t used_msgs = 0;

    for (const rosbag::MessageInstance &m : view) {
      sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
      if (!img_msg) {
        continue;
      }

      cv::Mat img_gray = decode_grayscale(img_msg);
      if (img_gray.empty()) {
        continue;
      }

      if (!tracker_initialized) {
        std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
        auto cam = std::make_shared<ov_core::CamRadtan>(img_gray.cols, img_gray.rows);
        Eigen::Matrix<double, 8, 1> calib;
        calib << 300.0, 300.0, static_cast<double>(img_gray.cols) * 0.5, static_cast<double>(img_gray.rows) * 0.5, 0.0, 0.0, 0.0, 0.0;
        cam->set_value(calib);
        cameras.insert({0, cam});

        ov_lightglue::TrackSuperLightGlueConfig cfg;
        cfg.superpoint_onnx_path = superpoint_onnx_path;
        cfg.lightglue_onnx_path = lightglue_onnx_path;
        cfg.use_gpu = use_gpu;
        cfg.max_keypoints = 1024;
        cfg.detect_min_confidence = -1.0f;
        cfg.match_min_confidence = -1.0f;

        tracker.reset(new ov_lightglue::TrackSuperLightGlue(cameras, 300, 0, false, ov_core::TrackBase::HistogramMethod::NONE, cfg));
        tracker_initialized = true;
      }

      if (!writer_initialized) {
        writer.open(output_mp4, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), out_fps, img_gray.size(), true);
        if (!writer.isOpened()) {
          std::cerr << "Failed to open output writer: " << output_mp4 << std::endl;
          return 4;
        }
        writer_initialized = true;
      }

      ov_core::CameraData data;
      data.timestamp = img_msg->header.stamp.toSec();
      data.sensor_ids.push_back(0);
      data.images.push_back(img_gray);
      data.masks.push_back(cv::Mat::zeros(img_gray.rows, img_gray.cols, CV_8UC1));
      tracker->feed_new_camera(data);

      const auto ids_by_cam = tracker->get_last_ids();
      const auto obs_by_cam = tracker->get_last_obs();
      const std::vector<size_t> curr_ids = (ids_by_cam.count(0) > 0) ? ids_by_cam.at(0) : std::vector<size_t>{};
      const std::vector<cv::KeyPoint> curr_kpts = (obs_by_cam.count(0) > 0) ? obs_by_cam.at(0) : std::vector<cv::KeyPoint>{};
      size_t carried = 0;
      for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
        if (prev_pts_by_id.find(curr_ids[i]) != prev_pts_by_id.end()) {
          ++carried;
        }
      }
      const size_t active = curr_ids.size();
      const size_t fresh = (active >= carried) ? (active - carried) : 0;
      metrics << frame_idx << "," << data.timestamp << "," << active << "," << carried << "," << fresh << "\n";

      cv::Mat frame_bgr;
      cv::cvtColor(img_gray, frame_bgr, cv::COLOR_GRAY2BGR);
      draw_overlay(frame_bgr, curr_ids, curr_kpts, prev_pts_by_id, frame_idx);
      writer.write(frame_bgr);

      prev_pts_by_id.clear();
      prev_pts_by_id.reserve(curr_ids.size());
      for (size_t i = 0; i < curr_ids.size() && i < curr_kpts.size(); ++i) {
        prev_pts_by_id[curr_ids[i]] = curr_kpts[i].pt;
      }

      ++frame_idx;
      ++used_msgs;
      if (frame_idx % 50 == 0) {
        std::cout << "processed_frames=" << frame_idx << std::endl;
      }
    }

    if (used_msgs == 0) {
      std::cerr << "No decodable sensor_msgs/Image frames on topic: " << topic << std::endl;
      return 3;
    }

    writer.release();
    metrics.close();
    bag.close();

    std::cout << "Done. frames_written=" << frame_idx << " output=" << output_mp4 << " metrics=" << output_csv << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "track_superlightglue_bag_to_mp4 failed: " << e.what() << std::endl;
    return 5;
  }

  return 0;
}
