cmake_minimum_required(VERSION 2.6)

project(LSD)

# Verbose make
#SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(CMAKE_BUILD_TYPE DEBUG)

# Library type (STATIC or SHARED)
SET(LIBRARY_TYPE STATIC)

# Set link and compile flags
# c++11 for chrono 
# -no-write-strings to remove deprecated conversion for the c-code.
# -pg for profiling the code. REMOVE WHEN DELIVERING  
# -O3 optimization flag  // + -ffast-math ?
SET(CXX_COVERAGE_FLAGS	"-std=c++11 -Wno-write-strings -O3 -pg")
SET(CXX_COVERAGE_LINK_FLAGS "")
SET(CMAKE_CXX_FLAGS	"${CMAKE_CXX_FLAGS} ${CXX_COVERAGE_FLAGS}")
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COVERAGE_LINK_FLAGS}")

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

subdirs(src)
subdirs(examples)
subdirs(tests)
