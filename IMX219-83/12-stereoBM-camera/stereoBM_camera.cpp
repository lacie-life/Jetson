#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>


#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

using namespace cv;
using std::cout; using std::cerr; using std::endl;

static void print_help()
{
    cout << endl << "Demo stereo matching converting L and R images into disparity and point clouds" << endl;
    cout << endl <<  "Usage: stereo_match [--blocksize=<block_size>]" << endl <<
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]" << endl <<
           "[--no-display]" << endl;
}

void F_Gray2Color(CvMat* gray_mat, CvMat* color_mat);
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, double Disp2Depth);

int main(int argc, char** argv) 
{    
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";

    int SADWindowSize, numberOfDisparities;
    bool no_display;
    float scale;
    double Disp2Depth = 0;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}{help h||}{max-disparity|0|}{blocksize|0|}{no-display||}{scale|1|}{i||}{e||}");
    if(parser.has("help"))
    {
        print_help();
        return 0;
    }
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));

    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        cout << "Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16" << endl;
        print_help();
        return -1;
    }
    if (scale < 0)
    {
        cout << "Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number" << endl;
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        cout << "Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number" << endl;
        return -1;
    }
    if( img1_filename.empty() || img2_filename.empty() )
    {
        cout << "Command-line parameter error: both left and right images must be specified" << endl;
        return -1;
    }
    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        cout << "Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)" << endl;
        return -1;
    }

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

    // Rectify two images with calibrated parameters
    Mat M1, D1, M2, D2;
    Mat R, T, R1, P1, R2, P2;
    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            cout << "Failed to open file " << intrinsic_filename.c_str() << endl;
            return -1;
        }
        
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            cout << "Failed to open file " << extrinsic_filename.c_str() << endl;
            return -1;
        }
        
        fs["R"] >> R;
        fs["T"] >> T;
    }

    while(1)
    {
	    Mat frame0, frame1;
        cam0 >> frame0;
        cam1 >> frame1;

        Mat imgLeft,imgRight;

        int color_mode = 0;
        if (!color_mode){
            cvtColor(frame0, imgLeft, COLOR_RGB2GRAY);
            cvtColor(frame1, imgRight, COLOR_RGB2GRAY);
        }
        else {
            imgLeft = frame0;
            imgRight = frame1;
        }

        if (imgLeft.empty())
        {
            cout << "Command-line parameter error: could not read the left camera" << endl;
            return -1;
        }
        if (imgRight.empty())
        {
            cout << "Command-line parameter error: could not read the right camera" << endl;
            return -1;
        }

        if (scale != 1.f)
        {
            Mat temp1, temp2;
            int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
            resize(imgLeft, temp1, Size(), scale, scale, method);
            imgLeft = temp1;
            resize(imgRight, temp2, Size(), scale, scale, method);
            imgRight = temp2;
        }

        Size img_size = imgLeft.size();

        // Rectify the images
        Rect roi1, roi2;
        Mat Q;
        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
        Disp2Depth = Q.at<double>(2, 3) * (-T.at<double>(0, 0));
        Mat img1r, img2r;
        remap(imgLeft, img1r, map11, map12, INTER_LINEAR);
        remap(imgRight, img2r, map21, map22, INTER_LINEAR);
        imgLeft = img1r;
        imgRight = img2r;

        // Set common parameters
        numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
        bm->setROI1(roi1);
        bm->setROI2(roi2);
        bm->setPreFilterCap(31);
        bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
        bm->setMinDisparity(0);
        bm->setNumDisparities(numberOfDisparities);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(15);
        bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        bm->setDisp12MaxDiff(1);

        // Generate the disparity map
        Mat disp, disp8;
        int64 t = getTickCount();
        bm->compute(imgLeft, imgRight, disp);
        t = getTickCount() - t;
        cout << "Time elapsed: " << t*1000/getTickFrequency() << "ms" << endl; 
    
        // Revise the disparity map to a suitable format
        double minVal; double maxVal;
        minMaxLoc( disp, &minVal, &maxVal );
        add(disp, -minVal, disp);
        divide(disp, 16, disp);
        disp.convertTo( disp, CV_8UC1);
        disp.convertTo( disp8, CV_8UC1, 255/numberOfDisparities);

        Mat dispRGB0(img_size.height, img_size.width, CV_8UC3);       
        CvMat cdisp, cdispRGB = dispRGB0;
        cdisp = disp8;
        F_Gray2Color(&cdisp, &cdispRGB);
        Mat dispRGB(cdispRGB.rows, cdispRGB.cols, CV_8UC3, cdispRGB.data.i);
        
        // Generate depth map
        Mat depth(imgLeft.size(), CV_16UC1);
        disp2Depth(disp, depth, Disp2Depth);
        // Generate cloud points
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);

        if( !no_display )
        {
            namedWindow("left", 1);
            imshow("left", imgLeft);
            namedWindow("right", 1);
            imshow("right", imgRight);
            namedWindow("disparity", 0);
            imshow("disparity", dispRGB);
            imshow("depth", depth);
            fflush(stdout);
        }


	    if((char)waitKey(30) == 27)
		    break;
    }
    cam0.release();
    cam1.release();
    return 0;
}


void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, double Disp2Depth)
{
    int type = dispMap.type();

    if (type == CV_8U)
    {
        int height = dispMap.rows;
        int width = dispMap.cols;

        uchar* dispData = (uchar*)dispMap.data;
        ushort* depthData = (ushort*)depthMap.data;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int id = i*width + j;
                if (!dispData[id]) {
                    depthData[id] = 65535;
                    continue;
                } 
                depthData[id] = ushort( (float)Disp2Depth/ ((float)dispData[id]) );
            }
        }
    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}

void F_Gray2Color(CvMat* gray_mat, CvMat* color_mat)
{
    if(color_mat)
        cvZero(color_mat);
		
    int stype = CV_MAT_TYPE(gray_mat->type), dtype = CV_MAT_TYPE(color_mat->type);
    int rows = gray_mat->rows, cols = gray_mat->cols;

    if (CV_ARE_SIZES_EQ(gray_mat, color_mat) && stype == CV_8UC1 && dtype == CV_8UC3)
    {
        CvMat* red = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
        CvMat* green = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
        CvMat* blue = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
        CvMat* mask = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);

        cvSubRS(gray_mat, cvScalar(255), blue);	
        cvCopy(gray_mat, red);			
        cvCopy(gray_mat, green);			
        cvCmpS(green, 128, mask, CV_CMP_GE );	
        cvSubRS(green, cvScalar(255), green, mask);
        cvConvertScale(green, green, 2.0, 0.0);
        cvMerge(blue, green, red, NULL, color_mat);

        cvReleaseMat( &red );
        cvReleaseMat( &green );
        cvReleaseMat( &blue );
        cvReleaseMat( &mask );
    }
}