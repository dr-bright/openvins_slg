cmake_minimum_required(VERSION 3.3)
project(ov_lightglue)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
endif ()
find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)

option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin QUIET COMPONENTS roscpp rosbag ov_core ov_init ov_eval)
find_package(ament_cmake QUIET)

if (catkin_FOUND AND ENABLE_ROS)
  catkin_package(
    INCLUDE_DIRS src
    CATKIN_DEPENDS roscpp rosbag ov_core ov_init ov_eval
  )
  include_directories(
    src
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )

  add_library(${PROJECT_NAME}
    src/track/slg_backend.cpp
    src/track/TrackSuperLightGlue.cpp
  )
  target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(slg_backend_smoke_test
    tests/slg_backend_smoke_test.cpp
  )
  target_link_libraries(slg_backend_smoke_test
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(slg_backend_smoke_test_vis
    tests/slg_backend_smoke_test_vis.cpp
  )
  target_link_libraries(slg_backend_smoke_test_vis
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
  )

  add_executable(onnxrt_check
    tests/onnxrt_check.cpp
  )
  target_link_libraries(onnxrt_check
    ${ONNXRUNTIME_LIB}
  )
elseif (ament_cmake_FOUND AND ENABLE_ROS)
  ament_package()
else ()
  message(STATUS "No ROS environment detected for ov_lightglue scaffold.")
endif ()
