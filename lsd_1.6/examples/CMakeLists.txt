cmake_minimum_required(VERSION 2.6)

# Add Libraries
include_directories(${LSD_SOURCE_DIR}/libs)
include_directories(${LSD_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

add_executable(lsd_call_example lsd_call_example.cpp)
target_link_libraries(lsd_call_example lsd)

add_executable(lsd_cmd lsd_cmd.cpp)
target_link_libraries(lsd_cmd lsd)

add_executable(lsd_opencv_example lsd_opencv_example.cpp)
target_link_libraries(lsd_opencv_example lsd_opencv ${OpenCV_LIBS})

add_executable(lsd_wrap_cmd lsd_wrap_cmd.cpp)
target_link_libraries(lsd_wrap_cmd lsd_wrap ${OpenCV_LIBS})

add_executable(lsd_opencv_cmd lsd_opencv_cmd.cpp)
target_link_libraries(lsd_opencv_cmd lsd_opencv ${OpenCV_LIBS})

add_executable(lsd_lines lsd_lines.cpp)
target_link_libraries(lsd_lines lsd_opencv ${OpenCV_LIBS})

add_executable(lsd_filter lsd_filterAngle.cpp)
target_link_libraries(lsd_filter lsd_opencv ${OpenCV_LIBS})
