#Build:
$mkdir build
$cd build
$cmake ..
$make

#Run:
$./stereoSGBM ../images/left1.jpg ../images/right1.jpg -i=../cal_results/intrinsics.yml -e=../cal_results/extrinsics.yml --max-disparity=64 --blocksize=11




