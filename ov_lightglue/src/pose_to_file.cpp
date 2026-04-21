/*
 * OpenVINS pose recorder (local copy in ov_lightglue).
 * Supports PoseWithCovarianceStamped, PoseStamped, TransformStamped, Odometry, PointStamped.
 */

#include <fstream>
#include <string>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "utils/print.h"

namespace ov_lightglue {

class Recorder {
public:
  explicit Recorder(const std::string &filename) {
    boost::filesystem::path dir(filename.c_str());
    if (boost::filesystem::create_directories(dir.parent_path())) {
      ROS_INFO("Created folder path to output file.");
      ROS_INFO("Path: %s", dir.parent_path().c_str());
    }
    if (boost::filesystem::exists(filename)) {
      ROS_WARN("Output file exists, deleting old file....");
      boost::filesystem::remove(filename);
    }

    outfile_.open(filename.c_str());
    if (outfile_.fail()) {
      ROS_ERROR("Unable to open output file!!");
      ROS_ERROR("Path: %s", filename.c_str());
      std::exit(EXIT_FAILURE);
    }

    outfile_ << "# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33" << std::endl;

    timestamp_ = -1;
    q_ItoG_ << 0, 0, 0, 1;
    p_IinG_ = Eigen::Vector3d::Zero();
    cov_rot_ = Eigen::Matrix<double, 3, 3>::Zero();
    cov_pos_ = Eigen::Matrix<double, 3, 3>::Zero();
    has_covariance_ = false;
  }

  void callback_odometry(const nav_msgs::OdometryPtr &msg) {
    timestamp_ = msg->header.stamp.toSec();
    q_ItoG_ << msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w;
    p_IinG_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    cov_pos_ << msg->pose.covariance.at(0), msg->pose.covariance.at(1), msg->pose.covariance.at(2), msg->pose.covariance.at(6),
        msg->pose.covariance.at(7), msg->pose.covariance.at(8), msg->pose.covariance.at(12), msg->pose.covariance.at(13),
        msg->pose.covariance.at(14);
    cov_rot_ << msg->pose.covariance.at(21), msg->pose.covariance.at(22), msg->pose.covariance.at(23), msg->pose.covariance.at(27),
        msg->pose.covariance.at(28), msg->pose.covariance.at(29), msg->pose.covariance.at(33), msg->pose.covariance.at(34),
        msg->pose.covariance.at(35);
    has_covariance_ = true;
    write();
  }

  void callback_pose(const geometry_msgs::PoseStampedPtr &msg) {
    timestamp_ = msg->header.stamp.toSec();
    q_ItoG_ << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w;
    p_IinG_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    has_covariance_ = false;
    write();
  }

  void callback_posecovariance(const geometry_msgs::PoseWithCovarianceStampedPtr &msg) {
    timestamp_ = msg->header.stamp.toSec();
    q_ItoG_ << msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w;
    p_IinG_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    cov_pos_ << msg->pose.covariance.at(0), msg->pose.covariance.at(1), msg->pose.covariance.at(2), msg->pose.covariance.at(6),
        msg->pose.covariance.at(7), msg->pose.covariance.at(8), msg->pose.covariance.at(12), msg->pose.covariance.at(13),
        msg->pose.covariance.at(14);
    cov_rot_ << msg->pose.covariance.at(21), msg->pose.covariance.at(22), msg->pose.covariance.at(23), msg->pose.covariance.at(27),
        msg->pose.covariance.at(28), msg->pose.covariance.at(29), msg->pose.covariance.at(33), msg->pose.covariance.at(34),
        msg->pose.covariance.at(35);
    has_covariance_ = true;
    write();
  }

  void callback_transform(const geometry_msgs::TransformStampedPtr &msg) {
    timestamp_ = msg->header.stamp.toSec();
    q_ItoG_ << msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z, msg->transform.rotation.w;
    p_IinG_ << msg->transform.translation.x, msg->transform.translation.y, msg->transform.translation.z;
    has_covariance_ = false;
    write();
  }

  void callback_point(const geometry_msgs::PointStampedPtr &msg) {
    timestamp_ = msg->header.stamp.toSec();
    p_IinG_ << msg->point.x, msg->point.y, msg->point.z;
    // Requested placeholder orientation for PointStamped output.
    q_ItoG_ << 1.0, 0.0, 0.0, 0.0;
    has_covariance_ = false;
    write();
  }

private:
  void write() {
    outfile_.precision(5);
    outfile_.setf(std::ios::fixed, std::ios::floatfield);
    outfile_ << timestamp_ << " ";

    outfile_.precision(6);
    outfile_ << p_IinG_.x() << " " << p_IinG_.y() << " " << p_IinG_.z() << " " << q_ItoG_(0) << " " << q_ItoG_(1) << " "
             << q_ItoG_(2) << " " << q_ItoG_(3);

    if (has_covariance_) {
      outfile_.precision(10);
      outfile_ << " " << cov_rot_(0, 0) << " " << cov_rot_(0, 1) << " " << cov_rot_(0, 2) << " " << cov_rot_(1, 1) << " "
               << cov_rot_(1, 2) << " " << cov_rot_(2, 2) << " " << cov_pos_(0, 0) << " " << cov_pos_(0, 1) << " "
               << cov_pos_(0, 2) << " " << cov_pos_(1, 1) << " " << cov_pos_(1, 2) << " " << cov_pos_(2, 2) << std::endl;
    } else {
      outfile_ << std::endl;
    }
  }

  std::ofstream outfile_;

  bool has_covariance_ = false;
  double timestamp_;
  Eigen::Vector4d q_ItoG_;
  Eigen::Vector3d p_IinG_;
  Eigen::Matrix<double, 3, 3> cov_rot_;
  Eigen::Matrix<double, 3, 3> cov_pos_;
};

} // namespace ov_lightglue

int main(int argc, char **argv) {
  ros::init(argc, argv, "pose_to_file");
  ros::NodeHandle nh("~");

  std::string verbosity;
  nh.param<std::string>("verbosity", verbosity, "INFO");
  ov_core::Printer::setPrintLevel(verbosity);

  std::string topic, topic_type, fileoutput;
  nh.getParam("topic", topic);
  nh.getParam("topic_type", topic_type);
  nh.getParam("output", fileoutput);

  PRINT_DEBUG("Done reading config values");
  PRINT_DEBUG(" - topic = %s", topic.c_str());
  PRINT_DEBUG(" - topic_type = %s", topic_type.c_str());
  PRINT_DEBUG(" - file = %s", fileoutput.c_str());

  ov_lightglue::Recorder recorder(fileoutput);

  ros::Subscriber sub;
  if (topic_type == std::string("PoseWithCovarianceStamped")) {
    sub = nh.subscribe(topic, 9999, &ov_lightglue::Recorder::callback_posecovariance, &recorder);
  } else if (topic_type == std::string("PoseStamped")) {
    sub = nh.subscribe(topic, 9999, &ov_lightglue::Recorder::callback_pose, &recorder);
  } else if (topic_type == std::string("TransformStamped")) {
    sub = nh.subscribe(topic, 9999, &ov_lightglue::Recorder::callback_transform, &recorder);
  } else if (topic_type == std::string("Odometry")) {
    sub = nh.subscribe(topic, 9999, &ov_lightglue::Recorder::callback_odometry, &recorder);
  } else if (topic_type == std::string("PointStamped")) {
    sub = nh.subscribe(topic, 9999, &ov_lightglue::Recorder::callback_point, &recorder);
  } else {
    PRINT_ERROR("The specified topic type is not supported");
    PRINT_ERROR("topic_type = %s", topic_type.c_str());
    PRINT_ERROR("please select from: PoseWithCovarianceStamped, PoseStamped, TransformStamped, Odometry, PointStamped");
    std::exit(EXIT_FAILURE);
  }

  ros::spin();
  return EXIT_SUCCESS;
}
