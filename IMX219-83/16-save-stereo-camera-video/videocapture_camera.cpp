#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using std::cout; using std::cerr; using std::endl;

int main(int, char**)
{
    Mat frame0 ,frame1;
    cout << "Opening camera..." << endl;
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    if(!cam0.isOpened())
    {
       cout << "cam0 is not opened." << endl;
       return -1;
    }
    if(!cam1.isOpened())
    {
       cout << "cam1 is not opened." << endl;
       return -1;
    }

    cout << "Frame width: " << cam0.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "     height: " << cam0.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Capturing FPS: " << cam0.get(CAP_PROP_FPS) << endl;

    cout << endl << "Press 'ESC' to quit, 'r' to record the video" << endl;
    cout << endl << "Start grabbing..." << endl;

    char file_dst[2][100];
    VideoWriter writer_l, writer_r;
    Size size;
    int fps = cam0.get(CAP_PROP_FPS);

    
    bool enableProcessing = false;
    int64 t0 = cv::getTickCount();
    int64 processingTime = 0;
    bool recording = false;
    for (;;)
    {
        cam0 >> frame0; // read the next frame from camera
        cam1 >> frame1;
        if (frame0.empty())
        {
            cerr << "ERROR: Can't grab left camera frame." << endl;
            break;
        }
        if (frame1.empty())
        {
            cerr << "ERROR: Can't grab right camera frame." << endl;
            break;
        }


        imshow("left", frame0);
        imshow("right", frame1);

        int key = waitKey(30);
        if (key == 27/*ESC*/)
            break;

        if (key == 'r'){
            recording = true; 
            
            size = frame0.size();

            writer_l.open("../videos/left.AVI", writer_l.fourcc('M', 'J', 'P', 'G'), fps, size, true);//CAP_OPENCV_MJPEG
            writer_r.open("../videos/right.AVI", writer_r.fourcc('M', 'J', 'P', 'G'), fps, size, true);//CAP_OPENCV_MJPEG
        }
        if (recording){
            writer_l.write(frame0);
            writer_r.write(frame1);
            waitKey(30);
        }
    }

    cam0.release();
    cam1.release();
    return 0;
}
