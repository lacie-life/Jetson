#Build:
$mkdir build
$cd build
$cmake ..
$make

#Run:
$./stereoBM_gpu_camera -i=../cal_results/intrinsics.yml -e=../cal_results/extrinsics.yml --blocksize=11 --ndisp=64




