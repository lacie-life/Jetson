cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(test-gpu)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

# Modern CMake
if(USE_CUDA)
    enable_language("CUDA")
    add_executable(test-gpu gpu-test.cu)
else(USE_CUDA)
    add_executable(test-gpu gpu-test.cu)
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()