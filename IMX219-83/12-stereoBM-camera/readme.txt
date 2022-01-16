#Build:
$mkdir build
$cd build
$cmake ..
$make

#Run:
$./stereoBM_camera -i=../cal_results/intrinsics.yml -e=../cal_results/extrinsics.yml --max-disparity=64 --blocksize=11




