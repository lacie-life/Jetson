cmake_minimum_required(VERSION 3.10)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
   project(vilib-test LANGUAGES CXX CUDA)
else()
   project(vilib-test LANGUAGES CXX)
endif()

find_package(vilib REQUIRED)
find_package(CUDA REQUIRED)
# only necessary if you happen to use opencv
find_package(OpenCV COMPONENTS core imgproc features2d highgui)

message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

file(GLOB_RECURSE VILIB_TEST_SOURCES
  src/*.cpp
  src/*.cu
  )

add_executable(vilib_tests ${VILIB_TEST_SOURCES})
include_directories(include)

target_link_libraries(vilib_tests
  vilib::vilib
  opencv_core opencv_imgproc opencv_features2d opencv_highgui
  ${CUDA_LIBRARIES})

install(TARGETS vilib_tests
  DESTINATION lib)