#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() 
{    
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    if(!cam0.isOpened())
    {
       printf("cam0 is not opened.\n");
       return -1;
    }
    if(!cam1.isOpened())
    {
       printf("cam1 is not opened.\n");
       return -1;
    }

    while(1)
    {
	    Mat frame0, frame1;
        cam0 >> frame0;
        cam1 >> frame1;
        //-- 1. Read the images
        Mat imgLeft,imgRight;
        cvtColor(frame0, imgRight, COLOR_RGB2GRAY);
        cvtColor(frame1, imgLeft, COLOR_RGB2GRAY);
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
        namedWindow("imgRight", WINDOW_AUTOSIZE);
        imshow("imgLeft", imgLeft);
        imshow("imgRight", imgRight);
        imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
        namedWindow("windowDisparity", WINDOW_NORMAL );
        imshow("windowDisparity", imgDisparity8U );

	    if((char)waitKey(30) == 27)
		    break;
    }

    return 0;
}
