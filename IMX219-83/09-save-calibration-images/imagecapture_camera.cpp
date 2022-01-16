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
       cout << "cam0 is not opened." << endl;;
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
    cout << endl << "Press 'ESC' or 'q' to quit" << endl;
    cout << endl << "Start grabbing..." << endl;

    char file_dst[2][100];
    int idx = 0;

    for (;;)
    {
        cam0 >> frame0; // read the next frame from the left camera
        cam1 >> frame1; // read the next frame from the right camera
        if (frame0.empty() || frame1.empty())
        {
            cerr << "ERROR: Can't grab camera frame." << endl;
            break;
        }

        int key = waitKey(30);
        if (key == 27 /*ESC*/ || key == 'q')
            break;
        if (key == 's')
        {    
            sprintf(file_dst[0], "../images/left%d.jpg", idx);
            sprintf(file_dst[1], "../images/right%d.jpg", idx);
            cv::imwrite(file_dst[0], frame0);
            cv::imwrite(file_dst[1], frame1);
            idx++;

            cout << "Images saved!" << endl;
        }
        
        // Merge the two images together and show
        Mat merge(frame0.rows,frame0.cols + frame1.cols + 1, frame0.type());
        frame0.colRange(0, frame0.cols).copyTo(merge.colRange(0, frame0.cols));
        frame1.colRange(0, frame1.cols).copyTo(merge.colRange(frame0.cols + 1, merge.cols));
        
        imshow("Merge",merge);
    }

    cam0.release();
    cam1.release();
    return 0;
}
