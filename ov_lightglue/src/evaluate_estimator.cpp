/*
 * ROS1 estimator evaluation node for OpenVINS / ov_lightglue runtime topics.
 * Produces CSV metrics and/or MP4 visualization overlays depending on params.
 */

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct LoopFeatPoint {
  std::size_t id = 0;
  cv::Point2f uv = cv::Point2f(0.0f, 0.0f);
  cv::Vec3f p3 = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

class EvaluateEstimatorNode {
public:
  explicit EvaluateEstimatorNode(const ros::NodeHandle &nh) : nh_(nh) {
    nh_.param<bool>("enable_csv", enable_csv_, true);
    nh_.param<bool>("enable_viz", enable_viz_, false);

    nh_.param<std::string>("output_csv", output_csv_path_, std::string("/tmp/ov_metrics.csv"));
    nh_.param<std::string>("output_mp4", output_mp4_path_, std::string("/tmp/ov_estimator_viz.mp4"));

    nh_.param<std::string>("topic_pose", topic_pose_, std::string("/ov_msckf/poseimu"));
    nh_.param<std::string>("topic_odom", topic_odom_, std::string("/ov_msckf/odomimu"));
    nh_.param<std::string>("topic_points_slam", topic_points_slam_, std::string("/ov_msckf/points_slam"));
    nh_.param<std::string>("topic_points_msckf", topic_points_msckf_, std::string("/ov_msckf/points_msckf"));
    nh_.param<std::string>("topic_points_aruco", topic_points_aruco_, std::string("/ov_msckf/points_aruco"));
    nh_.param<std::string>("topic_pose_gt", topic_pose_gt_, std::string("/ov_msckf/posegt"));

    nh_.param<std::string>("topic_image", topic_image_, std::string("/ov_msckf/trackhist"));
    nh_.param<std::string>("topic_loop_feats", topic_loop_feats_, std::string("/ov_msckf/loop_feats"));

    nh_.param<double>("fps", fps_, 20.0);
    nh_.param<double>("max_gt_time_diff_sec", max_gt_time_diff_sec_, 0.05);
    nh_.param<double>("max_sync_time_diff_sec", max_sync_time_diff_sec_, 0.10);
    nh_.param<double>("slam_assoc_threshold_m", slam_assoc_threshold_m_, 0.10);

    if (!enable_csv_ && !enable_viz_) {
      throw std::runtime_error("evaluate_estimator: both enable_csv and enable_viz are false; nothing to do");
    }

    if (fps_ <= 0.0) {
      throw std::runtime_error("evaluate_estimator: fps must be > 0");
    }

    if (enable_csv_) {
      csv_.open(output_csv_path_);
      if (!csv_.is_open()) {
        throw std::runtime_error("evaluate_estimator: failed to open output csv: " + output_csv_path_);
      }
      csv_ << "wall_time_sec,state_time_sec,dt_state_sec,"
              "slam_count,msckf_count,aruco_count,"
              "speed_mps,pose_pos_cov_trace,pose_ori_cov_trace,gt_pos_err_m\n";
      ROS_INFO("evaluate_estimator: CSV enabled -> %s", output_csv_path_.c_str());
    }

    sub_pose_ = nh_.subscribe(topic_pose_, 200, &EvaluateEstimatorNode::callback_pose, this);
    sub_odom_ = nh_.subscribe(topic_odom_, 200, &EvaluateEstimatorNode::callback_odom, this);
    sub_points_slam_ = nh_.subscribe(topic_points_slam_, 100, &EvaluateEstimatorNode::callback_points_slam, this);
    sub_points_msckf_ = nh_.subscribe(topic_points_msckf_, 100, &EvaluateEstimatorNode::callback_points_msckf, this);
    sub_points_aruco_ = nh_.subscribe(topic_points_aruco_, 100, &EvaluateEstimatorNode::callback_points_aruco, this);
    sub_pose_gt_ = nh_.subscribe(topic_pose_gt_, 100, &EvaluateEstimatorNode::callback_pose_gt, this);

    if (enable_viz_) {
      sub_image_ = nh_.subscribe(topic_image_, 20, &EvaluateEstimatorNode::callback_image, this);
      sub_loop_feats_ = nh_.subscribe(topic_loop_feats_, 100, &EvaluateEstimatorNode::callback_loop_feats, this);
      ROS_INFO("evaluate_estimator: VIZ enabled -> %s", output_mp4_path_.c_str());
      ROS_INFO("evaluate_estimator: image=%s loop_feats=%s", topic_image_.c_str(), topic_loop_feats_.c_str());
    }

    ROS_INFO("evaluate_estimator: pose=%s odom=%s slam=%s msckf=%s aruco=%s gt=%s", topic_pose_.c_str(), topic_odom_.c_str(),
             topic_points_slam_.c_str(), topic_points_msckf_.c_str(), topic_points_aruco_.c_str(), topic_pose_gt_.c_str());
  }

private:
  static cv::Mat decode_bgr8(const sensor_msgs::ImageConstPtr &msg) {
    if (!msg) {
      return cv::Mat();
    }

    try {
      if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
        return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
      }
      if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
        cv::Mat rgb = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8)->image;
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
      }
      if (msg->encoding == sensor_msgs::image_encodings::MONO8 || msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
        cv::Mat mono = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8)->image;
        cv::Mat bgr;
        cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
      }
      if (msg->encoding == sensor_msgs::image_encodings::MONO16 || msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        cv::Mat mono16 = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16)->image;
        cv::Mat mono8;
        mono16.convertTo(mono8, CV_8U, 1.0 / 256.0);
        cv::Mat bgr;
        cv::cvtColor(mono8, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
      }
      return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (...) {
      return cv::Mat();
    }
  }

  static std::size_t cloud_count(const sensor_msgs::PointCloud2ConstPtr &msg) {
    return static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
  }

  static cv::Vec3f point32_to_vec3f(const geometry_msgs::Point32 &p) {
    return cv::Vec3f(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
  }

  static float l2_sq(const cv::Vec3f &a, const cv::Vec3f &b) {
    const float dx = a[0] - b[0];
    const float dy = a[1] - b[1];
    const float dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }

  std::set<std::size_t> compute_slam_ids(const std::vector<LoopFeatPoint> &loop_points, const std::vector<cv::Vec3f> &slam_points) const {
    std::set<std::size_t> slam_ids;
    if (loop_points.empty() || slam_points.empty()) {
      return slam_ids;
    }

    const float max_sq = static_cast<float>(slam_assoc_threshold_m_ * slam_assoc_threshold_m_);
    for (const auto &lp : loop_points) {
      float best = std::numeric_limits<float>::max();
      for (const auto &sp : slam_points) {
        const float d2 = l2_sq(lp.p3, sp);
        if (d2 < best) {
          best = d2;
        }
      }
      if (best <= max_sq) {
        slam_ids.insert(lp.id);
      }
    }
    return slam_ids;
  }

  void callback_points_slam(const sensor_msgs::PointCloud2ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    slam_count_ = cloud_count(msg);

    if (!enable_viz_) {
      return;
    }

    slam_points_.clear();
    slam_points_.reserve(slam_count_);
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      slam_points_.emplace_back(*it_x, *it_y, *it_z);
    }
  }

  void callback_points_msckf(const sensor_msgs::PointCloud2ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    msckf_count_ = cloud_count(msg);
  }

  void callback_points_aruco(const sensor_msgs::PointCloud2ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    aruco_count_ = cloud_count(msg);
  }

  void callback_odom(const nav_msgs::OdometryConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto &v = msg->twist.twist.linear;
    speed_mps_ = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  void callback_pose_gt(const geometry_msgs::PoseStampedConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    gt_time_sec_ = msg->header.stamp.toSec();
    gt_px_ = msg->pose.position.x;
    gt_py_ = msg->pose.position.y;
    gt_pz_ = msg->pose.position.z;
    have_gt_ = true;
  }

  void callback_pose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);

    const double t = msg->header.stamp.toSec();
    const double dt = (have_last_state_time_) ? (t - last_state_time_sec_) : std::numeric_limits<double>::quiet_NaN();
    last_state_time_sec_ = t;
    have_last_state_time_ = true;

    const auto &p = msg->pose.pose.position;
    const auto &cov = msg->pose.covariance;

    pose_pos_cov_trace_ = cov[0] + cov[7] + cov[14];
    pose_ori_cov_trace_ = cov[21] + cov[28] + cov[35];

    double gt_pos_err = std::numeric_limits<double>::quiet_NaN();
    if (have_gt_ && std::abs(t - gt_time_sec_) <= max_gt_time_diff_sec_) {
      const double dx = p.x - gt_px_;
      const double dy = p.y - gt_py_;
      const double dz = p.z - gt_pz_;
      gt_pos_err = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    if (enable_csv_) {
      csv_ << ros::Time::now().toSec() << "," << t << "," << dt << "," << slam_count_ << "," << msckf_count_ << "," << aruco_count_ << ","
           << speed_mps_ << "," << pose_pos_cov_trace_ << "," << pose_ori_cov_trace_ << "," << gt_pos_err << "\n";
      csv_.flush();
    }
  }

  void callback_loop_feats(const sensor_msgs::PointCloudConstPtr &msg) {
    if (!enable_viz_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    loop_stamp_sec_ = msg->header.stamp.toSec();
    loop_recv_sec_ = ros::Time::now().toSec();
    loop_points_.clear();
    loop_points_.reserve(msg->points.size());

    const std::size_t n = std::min(msg->points.size(), msg->channels.size());
    for (std::size_t i = 0; i < n; ++i) {
      if (msg->channels[i].values.size() < 5) {
        continue;
      }
      LoopFeatPoint p;
      p.uv = cv::Point2f(msg->channels[i].values[2], msg->channels[i].values[3]);
      p.id = static_cast<std::size_t>(std::llround(msg->channels[i].values[4]));
      p.p3 = point32_to_vec3f(msg->points[i]);
      loop_points_.push_back(p);
    }
    have_loop_points_ = true;
  }

  void callback_image(const sensor_msgs::ImageConstPtr &msg) {
    if (!enable_viz_) {
      return;
    }

    cv::Mat frame_bgr = decode_bgr8(msg);
    if (frame_bgr.empty()) {
      return;
    }

    if (!writer_initialized_) {
      writer_.open(output_mp4_path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_, frame_bgr.size(), true);
      if (!writer_.isOpened()) {
        ROS_ERROR("evaluate_estimator: failed to open output mp4: %s", output_mp4_path_.c_str());
        enable_viz_ = false;
        return;
      }
      writer_initialized_ = true;
    }

    std::vector<LoopFeatPoint> curr_loop_points;
    std::vector<cv::Vec3f> slam_points;
    std::size_t slam_count;
    std::size_t msckf_count;
    std::size_t aruco_count;
    double speed_mps;
    double pose_pos_cov_trace;
    double pose_ori_cov_trace;
    double loop_stamp_sec;
    double loop_recv_sec;
    bool have_loop_points;

    {
      std::lock_guard<std::mutex> lock(mtx_);
      curr_loop_points = loop_points_;
      slam_points = slam_points_;
      slam_count = slam_count_;
      msckf_count = msckf_count_;
      aruco_count = aruco_count_;
      speed_mps = speed_mps_;
      pose_pos_cov_trace = pose_pos_cov_trace_;
      pose_ori_cov_trace = pose_ori_cov_trace_;
      loop_stamp_sec = loop_stamp_sec_;
      loop_recv_sec = loop_recv_sec_;
      have_loop_points = have_loop_points_;
    }

    std::size_t active = 0;
    std::size_t carried = 0;
    std::size_t fresh = 0;
    std::size_t lost = 0;
    std::size_t slam_active = 0;

    const double now_sec = ros::Time::now().toSec();
    if (have_loop_points && std::abs(now_sec - loop_recv_sec) <= max_sync_time_diff_sec_) {
      const std::set<std::size_t> slam_ids = compute_slam_ids(curr_loop_points, slam_points);
      slam_active = slam_ids.size();

      std::unordered_map<std::size_t, cv::Point2f> curr_uv_by_id;
      curr_uv_by_id.reserve(curr_loop_points.size());

      for (const auto &p : curr_loop_points) {
        ++active;
        curr_uv_by_id[p.id] = p.uv;

        const auto it_prev = prev_uv_by_id_.find(p.id);
        const bool is_carried = (it_prev != prev_uv_by_id_.end());
        if (is_carried) {
          ++carried;
          cv::line(frame_bgr, it_prev->second, p.uv, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
          cv::circle(frame_bgr, p.uv, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        } else {
          ++fresh;
          cv::circle(frame_bgr, p.uv, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        }

        if (slam_ids.find(p.id) != slam_ids.end()) {
          cv::circle(frame_bgr, p.uv, 4, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
      }

      for (const auto &kv : prev_uv_by_id_) {
        if (curr_uv_by_id.find(kv.first) == curr_uv_by_id.end()) {
          ++lost;
          const cv::Point2f p = kv.second;
          cv::line(frame_bgr, cv::Point2f(p.x - 3.0f, p.y - 3.0f), cv::Point2f(p.x + 3.0f, p.y + 3.0f), cv::Scalar(0, 0, 120), 1,
                   cv::LINE_AA);
          cv::line(frame_bgr, cv::Point2f(p.x + 3.0f, p.y - 3.0f), cv::Point2f(p.x - 3.0f, p.y + 3.0f), cv::Scalar(0, 0, 120), 1,
                   cv::LINE_AA);
        }
      }

      prev_uv_by_id_.swap(curr_uv_by_id);
    }

    cv::putText(frame_bgr,
                "frame=" + std::to_string(frame_index_) + " active=" + std::to_string(active) + " carried=" + std::to_string(carried) +
                    " new=" + std::to_string(fresh) + " lost=" + std::to_string(lost) + " slam_active=" + std::to_string(slam_active),
                cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    cv::putText(frame_bgr,
                "slam3d=" + std::to_string(slam_count) + " msckf3d=" + std::to_string(msckf_count) +
                    " aruco3d=" + std::to_string(aruco_count) + " speed=" + cv::format("%.2f", speed_mps) +
                    "m/s cov_pos=" + cv::format("%.2e", pose_pos_cov_trace) + " cov_ori=" + cv::format("%.2e", pose_ori_cov_trace),
                cv::Point(10, 48), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    writer_.write(frame_bgr);
    ++frame_index_;
    if (frame_index_ % 100 == 0) {
      ROS_INFO("evaluate_estimator: frames_written=%zu", frame_index_);
    }
  }

private:
  ros::NodeHandle nh_;

  ros::Subscriber sub_pose_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_points_slam_;
  ros::Subscriber sub_points_msckf_;
  ros::Subscriber sub_points_aruco_;
  ros::Subscriber sub_pose_gt_;

  ros::Subscriber sub_image_;
  ros::Subscriber sub_loop_feats_;

  bool enable_csv_ = true;
  bool enable_viz_ = false;

  std::string output_csv_path_;
  std::string output_mp4_path_;

  std::string topic_pose_;
  std::string topic_odom_;
  std::string topic_points_slam_;
  std::string topic_points_msckf_;
  std::string topic_points_aruco_;
  std::string topic_pose_gt_;
  std::string topic_image_;
  std::string topic_loop_feats_;

  double fps_ = 20.0;
  double max_gt_time_diff_sec_ = 0.05;
  double max_sync_time_diff_sec_ = 0.10;
  double slam_assoc_threshold_m_ = 0.10;

  std::ofstream csv_;

  cv::VideoWriter writer_;
  bool writer_initialized_ = false;
  std::size_t frame_index_ = 0;

  std::mutex mtx_;

  std::size_t slam_count_ = 0;
  std::size_t msckf_count_ = 0;
  std::size_t aruco_count_ = 0;

  double speed_mps_ = std::numeric_limits<double>::quiet_NaN();
  double pose_pos_cov_trace_ = std::numeric_limits<double>::quiet_NaN();
  double pose_ori_cov_trace_ = std::numeric_limits<double>::quiet_NaN();

  bool have_last_state_time_ = false;
  double last_state_time_sec_ = 0.0;

  bool have_gt_ = false;
  double gt_time_sec_ = 0.0;
  double gt_px_ = 0.0;
  double gt_py_ = 0.0;
  double gt_pz_ = 0.0;

  std::vector<LoopFeatPoint> loop_points_;
  std::vector<cv::Vec3f> slam_points_;
  double loop_stamp_sec_ = -1.0;
  double loop_recv_sec_ = -1.0;
  bool have_loop_points_ = false;
  std::unordered_map<std::size_t, cv::Point2f> prev_uv_by_id_;
};

} // namespace

int main(int argc, char **argv) {
  ros::init(argc, argv, "evaluate_estimator");
  ros::NodeHandle nh("~");

  try {
    EvaluateEstimatorNode node(nh);
    ros::spin();
  } catch (const std::exception &e) {
    ROS_ERROR("evaluate_estimator: %s", e.what());
    return 1;
  }

  return 0;
}
