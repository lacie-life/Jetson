# YOLOv5 LibTorch
Real time object detection with deployment of YOLOv5 through LibTorch C++ API

### Environment

- Ubuntu 20.04
- OpenCV 4.2.0
- LibTorch 1.9.1 (cxx11 abi)
- CMake 3.10.2

### Getting Started

1. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

2. Install LibTorch.

   ```shell
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-1.9.1+cpu.zip
   ```

3. Edit "CMakeLists.txt" to configure OpenCV and LibTorch correctly.

4. Compile and run.

   ```shell
   cd build
   cmake ..
   make
   ./../bin/YOLOv5LibTorch
   ```

Note: COCO-pretrained YOLOv5s model has been provided. For more pretrained models, see [yolov5](https://github.com/ultralytics/yolov5).