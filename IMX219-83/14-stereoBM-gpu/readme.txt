#Build:
$mkdir build
$cd build
$cmake ..
$make

#Run:
$./stereoBM_gpu ../images/left1.jpg ../images/right1.jpg -i=../cal_results/intrinsics.yml -e=../cal_results/extrinsics.yml --blocksize=11 --ndisp=64





