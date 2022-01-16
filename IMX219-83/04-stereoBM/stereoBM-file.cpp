#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main( int argc, char** argv )
{
  if( argc != 3 )
  { 
    return -1;
  }

  //-- 1. Read the images
  Mat imgLeft = imread( argv[1], IMREAD_GRAYSCALE );
  Mat imgRight = imread( argv[2], IMREAD_GRAYSCALE );
  //-- And create the image in which we will save our disparities
  Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
  Mat imgDisparity8U = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );
  if( !imgLeft.data || !imgRight.data )
  { 
    std::cout<< " --(!) Error reading images " << std::endl; 
    return -1; 
  }
  //-- 2. Call the constructor for StereoBM
  int ndisparities = 16*5;   /**< Range of disparity */
  int SADWindowSize = 21; /**< Size of the block window. Must be odd */
  cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);
  //-- 3. Calculate the disparity image
  bm->compute(imgLeft, imgRight, imgDisparity16S);
  //-- Check its extreme values
  double minVal; double maxVal;
  minMaxLoc( imgDisparity16S, &minVal, &maxVal );
  printf("Min disp: %f Max value: %f \n", minVal, maxVal);
  //-- 4. Display it as a CV_8UC1 image
  namedWindow("imgLeft", WINDOW_AUTOSIZE);
  namedWindow("imgLeft", WINDOW_AUTOSIZE);
  imshow("imgLeft", imgLeft);
  imshow("imgRight", imgRight);
  imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
  namedWindow("Disparity", WINDOW_NORMAL );
  imshow("Disparity", imgDisparity8U );
  //-- 5. Save the image
  imwrite("stereoBM.jpg", imgDisparity16S);

  waitKey(0);
  return 0;
}


