#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "opencv4/opencv2/opencv.hpp"
#include <opencv4/opencv2/core/utility.hpp>
#include "opencv4/opencv2/cudastereo.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/calib3d/calib3d.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/core/core_c.h"
#include "opencv4/opencv2/core/types_c.h"

using namespace cv;
using std::cout; using std::cerr; using std::endl; using std::runtime_error;

static void printHelp()
{
    cout << "Usage: stereo_match\n"
        << "\t--left <left_view> --right <right_view>\n"
        << "\t--ndisp <number> # number of disparity levels\n"
        << "\t--block_size <number> # size of SAD window\n";
}

void F_Gray2Color(CvMat* gray_mat, CvMat* color_mat);
void disp2Depth(Mat dispMap, Mat &depthMap, double Disp2Depth);


int main(int argc, char** argv)
{
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";

    bool no_display = false;

    Mat left_src, right_src;
    Mat left, right;
    cuda::GpuMat d_left, d_right;

    int ndisp, SADWindowSize;
    double Disp2Depth = 0.0;
    float scale = 1;

    CommandLineParser parser(argc, argv, "{@arg1||}{@arg2||}{ndisp|0|}{i||}{e||}{scale|1|}{blocksize|0|}");

    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));

    ndisp = parser.get<int>("ndisp");
    SADWindowSize = parser.get<int>("blocksize");

    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");


    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        cout << "Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)" << endl;
        return -1;
    }

    // Load images
    left_src = imread(img1_filename);
    right_src = imread(img2_filename);
    if (left_src.empty()) throw runtime_error("can't open file \"" + img1_filename + "\"");
    if (right_src.empty()) throw runtime_error("can't open file \"" + img2_filename + "\"");
    cvtColor(left_src, left, COLOR_BGR2GRAY);
    cvtColor(right_src, right, COLOR_BGR2GRAY);

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(left, temp1, Size(), scale, scale, method);
        left = temp1;
        resize(right, temp2, Size(), scale, scale, method);
        right = temp2;
    }

    Size img_size = left.size();

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
        remap(left, img1r, map11, map12, INTER_LINEAR);
        remap(right, img2r, map21, map22, INTER_LINEAR);

        Disp2Depth = Q.at<double>(2, 3) * -T.at<double>(0, 0);

        left = img1r;
        right = img2r;
    }

    ndisp = ndisp > 0 ? ndisp : ((img_size.width/8) + 15) & -16;

    d_left.upload(left);
    d_right.upload(right);

    imshow("left", left);
    imshow("right", right);
  
    // Set common parameters
    Ptr<cuda::StereoBM> bm = cuda::createStereoBM(ndisp, SADWindowSize);

    // Prepare disparity map of specified type
    Mat disp(left.size(), CV_8U);
    Mat disp8(left.size(), CV_8U);
    cuda::GpuMat d_disp(left.size(), CV_8U);

    int64 t = getTickCount();
    bm->compute(d_left, d_right, d_disp);
    t = getTickCount() - t;
    cout << "Time elapsed: " << t*1000/getTickFrequency() << "ms" << endl; 
    
    d_disp.download(disp);

    // Revise the disparity image to a suitable format
    disp.convertTo( disp8, CV_8UC1, 255/(ndisp));
    
    
    Mat dispRGB0(img_size.height, img_size.width, CV_8UC3);       // 
    CvMat cdisp, cdispRGB = dispRGB0;
    cdisp = disp8;
    F_Gray2Color(&cdisp, &cdispRGB);
    Mat dispRGB(cdispRGB.rows, cdispRGB.cols, CV_8UC3, cdispRGB.data.i);

    // Generate depth image
    Mat depth(left.size(), CV_16UC1);
    disp2Depth(disp, depth, Disp2Depth);
    // Generate cloud points
    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q, true);
    
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", left_src);
        namedWindow("right", 1);
        imshow("right", right_src);
        namedWindow("disparity", 0);
        imshow("disparity", dispRGB);
        imshow("depth", depth);
        cout << "press any key to continue...";
        fflush(stdout);
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