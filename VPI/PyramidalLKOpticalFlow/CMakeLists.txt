cmake_minimum_required(VERSION 3.2)

project(vpi_PyrLK)

find_package(vpi REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_executable(vpi_PyrLK main.cpp)

target_link_libraries(vpi_PyrLK vpi opencv_core opencv_imgproc opencv_imgcodecs)
