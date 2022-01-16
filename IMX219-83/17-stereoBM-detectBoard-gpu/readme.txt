#Build:
$mkdir build
$cd build
$cmake ..
$make

#Run:
$./stereoBM_gpu ../images/left3.jpg ../images/right3.jpg -i=../cal_results/intrinsics.yml -e=../cal_results/extrinsics.yml --blocksize=11 --ndisp=64 -w=9 -h=6






