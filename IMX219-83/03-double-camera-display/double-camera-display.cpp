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
        imshow("cam0",frame0);
        imshow("cam1",frame1);
        if((char)waitKey(30) == 27)
            break;
    }
    
    return 0;
}
