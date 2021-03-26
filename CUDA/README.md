# CUDA

Complie command line:

nvcc -o add src/add.cu -gencode arch=compute_50,code=sm_50

Test command line:
nvprof ./excuta –unified-memory-profiling per-process-device

# CUDA Zero-Copy
https://developer.nvidia.com/blog/unified-memory-in-cuda-6/

https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

https://www.fastcompression.com/blog/jetson-zero-copy.htm

https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_CUDA_Memory_Management#NVIDIA_Jetson_AGX_Xavier

# CUDA for Tegra

https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html

# Course

https://kth.instructure.com/courses/12406
