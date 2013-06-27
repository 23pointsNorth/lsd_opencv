cmake_minimum_required(VERSION 2.6)

# Add Libraries
include_directories(${LSD_SOURCE_DIR}/libs)
include_directories(${LSD_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

add_executable(visual_test visual_test.cpp)
target_link_libraries(visual_test lsd_opencv lsd_wrap ${OpenCV_LIBS})

add_executable(houghlines_cmp houghlines.cpp)
target_link_libraries(houghlines_cmp ${OpenCV_LIBS})

add_executable(accuracy_test accuracy_test.cpp)
target_link_libraries(accuracy_test ${OpenCV_LIBS} lsd_opencv)
