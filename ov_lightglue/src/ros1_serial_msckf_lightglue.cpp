/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <memory>

#include "core/VioManagerSLG.h"
#include "core/VioManagerOptionsSLG.h"
#include "ros/ROS1VisualizerSLG.h"
#include "track/TrackerSuperLightGlueConfig.h"
#include "utils/dataset_reader.h"

using namespace ov_lightglue;
using namespace ov_msckf;

std::shared_ptr<VioManagerSLG> sys;
std::shared_ptr<ov_msckf::ROS1VisualizerSLG> viz;

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Launch our ros node
  ros::init(argc, argv, "ros1_serial_msckf_lightglue");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  ov_msckf::VioManagerOptionsSLG params;
  params.print_and_load(parser);
  // params.num_opencv_threads = 0; // uncomment if you want repeatability
  // params.use_multi_threading_pubs = 0; // uncomment if you want repeatability
  params.use_multi_threading_subs = false;

  TrackSuperLightGlueConfig tracker_config;
  parser->parse_config("slg_superpoint_onnx_path", tracker_config.superpoint_onnx_path);
  parser->parse_config("slg_lightglue_onnx_path", tracker_config.lightglue_onnx_path);
  parser->parse_config("slg_use_gpu", tracker_config.use_gpu);
  parser->parse_config("slg_max_keypoints", tracker_config.max_keypoints);
  parser->parse_config("slg_detect_min_confidence", tracker_config.detect_min_confidence);
  parser->parse_config("slg_match_min_confidence", tracker_config.match_min_confidence);
  parser->parse_config("slg_enable_temporal_ransac", tracker_config.enable_temporal_ransac);
  parser->parse_config("slg_temporal_ransac_min_matches", tracker_config.temporal_ransac_min_matches);
  parser->parse_config("slg_temporal_ransac_min_inliers", tracker_config.temporal_ransac_min_inliers);
  parser->parse_config("slg_temporal_ransac_threshold_px", tracker_config.temporal_ransac_threshold_px);
  parser->parse_config("slg_temporal_ransac_confidence", tracker_config.temporal_ransac_confidence);
  parser->parse_config("slg_min_feature_distance_px", tracker_config.min_feature_distance_px);
  parser->parse_config("slg_refill_tracks", tracker_config.refill_tracks);

  nh->param<std::string>("slg_superpoint_onnx_path", tracker_config.superpoint_onnx_path, tracker_config.superpoint_onnx_path);
  nh->param<std::string>("slg_lightglue_onnx_path", tracker_config.lightglue_onnx_path, tracker_config.lightglue_onnx_path);

  if (tracker_config.superpoint_onnx_path.empty() || tracker_config.lightglue_onnx_path.empty()) {
    PRINT_ERROR(RED "[SERIAL]: slg_superpoint_onnx_path and slg_lightglue_onnx_path must be configured\n" RESET);
    return EXIT_FAILURE;
  }

  sys = std::make_shared<ov_msckf::VioManagerSLG>(params, tracker_config);
  viz = std::make_shared<ov_msckf::ROS1VisualizerSLG>(nh, std::static_pointer_cast<ov_msckf::VioManagerSLG>(sys));

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Our imu topic
  std::string topic_imu;
  nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  PRINT_DEBUG("[SERIAL]: imu: %s\n", topic_imu.c_str());

  // Our camera topics
  std::vector<std::string> topic_cameras;
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string cam_topic;
    nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    topic_cameras.emplace_back(cam_topic);
    PRINT_DEBUG("[SERIAL]: cam: %s\n", cam_topic.c_str());
  }

  // Location of the ROS bag we want to read in
  std::string path_to_bag;
  nh->param<std::string>("path_bag", path_to_bag, "/home/patrick/datasets/eth/V1_01_easy.bag");
  PRINT_DEBUG("[SERIAL]: ros bag path is: %s\n", path_to_bag.c_str());

  // Load groundtruth if we have it
  // NOTE: needs to be a csv ASL format file
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;
  if (nh->hasParam("path_gt")) {
    std::string path_to_gt;
    nh->param<std::string>("path_gt", path_to_gt, "");
    if (!path_to_gt.empty()) {
      ov_core::DatasetReader::load_gt_file(path_to_gt, gt_states);
      PRINT_DEBUG("[SERIAL]: gt file path is: %s\n", path_to_gt.c_str());
    }
  }

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start, bag_durr;
  nh->param<double>("bag_start", bag_start, 0);
  nh->param<double>("bag_durr", bag_durr, -1);
  PRINT_DEBUG("[SERIAL]: bag start: %.1f\n", bag_start);
  PRINT_DEBUG("[SERIAL]: bag duration: %.1f\n", bag_durr);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  auto decode_image_msg = [](const rosbag::MessageInstance &msg) -> sensor_msgs::ImageConstPtr {
    sensor_msgs::ImageConstPtr img = msg.instantiate<sensor_msgs::Image>();
    if (img != nullptr) {
      return img;
    }

    sensor_msgs::CompressedImageConstPtr img_compressed = msg.instantiate<sensor_msgs::CompressedImage>();
    if (img_compressed == nullptr) {
      return nullptr;
    }

    cv::Mat compressed_buffer(static_cast<int>(img_compressed->data.size()), 1, CV_8UC1,
                              const_cast<uint8_t *>(img_compressed->data.data()));
    cv::Mat image = cv::imdecode(compressed_buffer, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      return nullptr;
    }

    return cv_bridge::CvImage(img_compressed->header, sensor_msgs::image_encodings::MONO8, image).toImageMsg();
  };

  // Load rosbag here, and find messages we can play
  rosbag::Bag bag;
  bag.open(path_to_bag, rosbag::bagmode::Read);

  // We should load the bag as a view
  // Here we go from beginning of the bag to the end of the bag
  rosbag::View view_full;
  rosbag::View view;

  // Start a few seconds in from the full view time
  // If we have a negative duration then use the full bag length
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  time_init += ros::Duration(bag_start);
  ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
  PRINT_DEBUG("time start = %.6f\n", time_init.toSec());
  PRINT_DEBUG("time end   = %.6f\n", time_finish.toSec());
  view.addQuery(bag, time_init, time_finish);

  // Check to make sure we have data to play
  if (view.size() == 0) {
    PRINT_ERROR(RED "[SERIAL]: No messages to play on specified topics.  Exiting.\n" RESET);
    ros::shutdown();
    return EXIT_FAILURE;
  }

  // We going to loop through and collect a list of all messages
  // This is done so we can access arbitrary points in the bag
  // NOTE: if we instantiate messages here, this requires the whole bag to be read
  // NOTE: thus we just check the topic which allows us to quickly loop through the index
  // NOTE: see this PR https://github.com/ros/ros_comm/issues/117
  double max_camera_time = -1;
  std::vector<rosbag::MessageInstance> msgs;
  for (const rosbag::MessageInstance &msg : view) {
    if (!ros::ok())
      break;
    if (msg.getTopic() == topic_imu) {
      // if (msg.instantiate<sensor_msgs::Imu>() == nullptr) {
      //   PRINT_ERROR(RED "[SERIAL]: IMU topic has unmatched message types!!\n" RESET);
      //   PRINT_ERROR(RED "[SERIAL]: Supports: sensor_msgs::Imu\n" RESET);
      //   return EXIT_FAILURE;
      // }
      msgs.push_back(msg);
    }
    for (int i = 0; i < params.state_options.num_cameras; i++) {
      if (msg.getTopic() == topic_cameras.at(i)) {
        if (decode_image_msg(msg) == nullptr) {
          PRINT_ERROR(RED "[SERIAL]: Image topic has unmatched message type or decode failure.\n" RESET);
          PRINT_ERROR(RED "[SERIAL]: Supports: sensor_msgs::Image and sensor_msgs::CompressedImage\n" RESET);
          return EXIT_FAILURE;
        }
        msgs.push_back(msg);
        max_camera_time = std::max(max_camera_time, msg.getTime().toSec());
      }
    }
  }
  PRINT_DEBUG("[SERIAL]: total of %zu messages!\n", msgs.size());

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Loop through our message array, and lets process them
  std::set<int> used_index;
  for (int m = 0; m < (int)msgs.size(); m++) {

    // End once we reach the last time, or skip if before beginning time (shouldn't happen)
    if (!ros::ok() || msgs.at(m).getTime() > time_finish || msgs.at(m).getTime().toSec() > max_camera_time)
      break;
    if (msgs.at(m).getTime() < time_init)
      continue;

    // Skip messages that we have already used
    if (used_index.find(m) != used_index.end()) {
      used_index.erase(m);
      continue;
    }

    // IMU processing
    if (msgs.at(m).getTopic() == topic_imu) {
      // PRINT_DEBUG("processing imu = %.3f sec\n", msgs.at(m).getTime().toSec() - time_init.toSec());
      viz->callback_inertial(msgs.at(m).instantiate<sensor_msgs::Imu>());
    }

    // Camera processing
    for (int cam_id = 0; cam_id < params.state_options.num_cameras; cam_id++) {

      // Skip if this message is not a camera topic
      if (msgs.at(m).getTopic() != topic_cameras.at(cam_id))
        continue;

      // We have a matching camera topic here, now find the other cameras for this time
      // For each camera, we will find the nearest timestamp (within 0.02sec) that is greater than the current
      // If we are unable, then this message should just be skipped since it isn't a sync'ed pair!
      std::map<int, int> camid_to_msg_index;
      double meas_time = msgs.at(m).getTime().toSec();
      for (int cam_idt = 0; cam_idt < params.state_options.num_cameras; cam_idt++) {
        if (cam_idt == cam_id) {
          camid_to_msg_index.insert({cam_id, m});
          continue;
        }
        int cam_idt_idx = -1;
        for (int mt = m; mt < (int)msgs.size(); mt++) {
          if (msgs.at(mt).getTopic() != topic_cameras.at(cam_idt))
            continue;
          if (std::abs(msgs.at(mt).getTime().toSec() - meas_time) < 0.02)
            cam_idt_idx = mt;
          break;
        }
        if (cam_idt_idx != -1) {
          camid_to_msg_index.insert({cam_idt, cam_idt_idx});
        }
      }

      // Skip processing if we were unable to find any messages
      if ((int)camid_to_msg_index.size() != params.state_options.num_cameras) {
        PRINT_DEBUG(YELLOW "[SERIAL]: Unable to find stereo pair for message %d at %.2f into bag (will skip!)\n" RESET, m,
                    meas_time - time_init.toSec());
        continue;
      }

      // Check if we should initialize using the groundtruth
      Eigen::Matrix<double, 17, 1> imustate;
      if (!gt_states.empty() && !sys->initialized() && ov_core::DatasetReader::get_gt_state(meas_time, imustate, gt_states)) {
        // biases are pretty bad normally, so zero them
        // imustate.block(11,0,6,1).setZero();
        sys->initialize_with_gt(imustate);
      }

      // Pass our data into our visualizer callbacks!
      // PRINT_DEBUG("processing cam = %.3f sec\n", msgs.at(m).getTime().toSec() - time_init.toSec());
      if (params.state_options.num_cameras == 1) {
        auto msg0 = decode_image_msg(msgs.at(camid_to_msg_index.at(0)));
        if (msg0 == nullptr) {
          PRINT_ERROR(RED "[SERIAL]: Failed to decode camera frame for cam0\n" RESET);
          return EXIT_FAILURE;
        }
        viz->callback_monocular(msg0, 0);
      } else if (params.state_options.num_cameras == 2) {
        auto msg0 = decode_image_msg(msgs.at(camid_to_msg_index.at(0)));
        auto msg1 = decode_image_msg(msgs.at(camid_to_msg_index.at(1)));
        if (msg0 == nullptr || msg1 == nullptr) {
          PRINT_ERROR(RED "[SERIAL]: Failed to decode stereo camera frame\n" RESET);
          return EXIT_FAILURE;
        }
        used_index.insert(camid_to_msg_index.at(0)); // skip this message
        used_index.insert(camid_to_msg_index.at(1)); // skip this message
        viz->callback_stereo(msg0, msg1, 0, 1);
      } else {
        PRINT_ERROR(RED "[SERIAL]: We currently only support 1 or 2 camera serial input....\n" RESET);
        return EXIT_FAILURE;
      }

      break;
    }
  }

  // Final visualization
  viz->visualize_final();

  // Done!
  return EXIT_SUCCESS;
}
