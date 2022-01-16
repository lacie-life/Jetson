#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() 
{
    VideoCapture cam("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    if(!cam.isOpened())
    {
       printf("Camera is not opened.\n");
       return -1;
    }

    while(1)
    {
        Mat frame;
        cam >> frame;
        imshow("original",frame);
        if((char)waitKey(30) == 27)
            break;
    }
    
    return 0;
}
