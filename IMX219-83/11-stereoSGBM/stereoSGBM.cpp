#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

#include <stdio.h>
#include <iostream>
using namespace cv;
using std::cout; using std::cerr; using std::endl;

void F_Gray2Color(CvMat* gray_mat, CvMat* color_mat);

void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, double Disp2Depth);

static void print_help()
{
    cout << endl << "Demo stereo matching converting L and R images into disparity and point clouds" << endl;;
    cout << endl << "Usage: stereo_match <left_image> <right_image> [--blocksize=<block_size>]" << endl << 
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]" << endl <<
           "[--no-display]" << endl;;
}

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

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,64,11);
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

    int color_mode = -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty())
    {
        cout << "Command-line parameter error: could not load the first input image file" << endl;
        return -1;
    }
    if (img2.empty())
    {
        cout << "Command-line parameter error: could not load the second input image file" << endl;
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();
    
    
    // Rectify two images with calibrated parameters
    Rect roi1, roi2;
    Mat Q;
    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            cout << "Failed to open file " << intrinsic_filename.c_str() << endl;
            return -1;
        }

        Mat M1, D1, M2, D2;
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

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        Disp2Depth = Q.at<double>(2, 3) * -T.at<double>(0, 0);

        img1 = img1r;
        img2 = img2r;
    }

    // Set common parameters
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img1.channels();   
    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);


    Mat disp(img1.size(), CV_16S), disp8(img1.size(), CV_8U);
    int64 t = getTickCount();
    sgbm->compute(img1, img2, disp);
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
    Mat depth(img1.size(), CV_16UC1);
    disp2Depth(disp, depth, Disp2Depth);

    // Generate cloud points
    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q, true);
        
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", img1);
        namedWindow("right", 1);
        imshow("right", img2);
        namedWindow("disparity", 0);
        imshow("disparity", dispRGB);
        imshow("depth", depth);
        cout << "press any key to continue...";
        waitKey();
        cout << endl;
    }
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