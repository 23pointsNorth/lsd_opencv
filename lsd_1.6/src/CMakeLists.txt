cmake_minimum_required(VERSION 2.6)

include_directories(${LSD_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

# Add source code properties
#set_source_files_properties(lsd.c lsd_call_example.c PROPERTIES LANGUAGE CXX)

add_library(lsd SHARED lsd.cpp)

add_library(lsd_opencv SHARED lsd_opencv.cpp)
target_link_libraries(lsd_opencv ${OpenCV_LIBS})

add_library(lsd_wrap SHARED lsd_wrap.cpp)
target_link_libraries(lsd_wrap lsd)
