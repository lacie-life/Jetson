#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <time.h>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Array.h>
#include <vpi/Status.h>

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/HarrisCorners.h>

 #define CHECK_STATUS(STMT)                                    \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);

static cv::Mat DrawKeypoints (cv::Mat img, VPIKeypoint *kpts, uint32_t *scores, int numKeypoints)
{
    cv::Mat out;
    img.convertTo(out, CV_8UC1);

    cvtColor(out, out, cv::COLOR_GRAY2BGR);

    if(numKeypoints == 0)
    {
        return out;
    }

    // prepare our colormap
    cv::Mat cmap(1, 256, CV_8UC3);
    {
        cv::Mat gray(1, 256, CV_8UC1);
        for (int i = 0; i < 256; ++i)
        {
            gray.at<unsigned char>(0, i) = i;
        }
        applyColorMap(gray, cmap, cv::COLORMAP_HOT);
    }
  
    float maxScore = *std::max_element(scores, scores + numKeypoints);
  
    for (int i = 0; i < numKeypoints; ++i)
    {
        cv::Vec3b color = cmap.at<cv::Vec3b>(scores[i] / maxScore * 255);
        circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, cv::Scalar(color[0], color[1], color[2]), -1);
    }
  
    return out;
}

int main(int argc, char *argv[])
{

    cv::Mat cvImage;

    VPIImage imgInput     = NULL;
    VPIImage imgGrayscale = NULL;
    VPIArray keypoints    = NULL;
    VPIArray scores       = NULL;
    VPIStream stream      = NULL;
    VPIPayload harris     = NULL;

    int retval = 0;

    try 
    {
        if(argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda> <input image>");
        }

        std::string strBackend    = argv[1];
        std::string strInputFileName = argv[2];

        // Now parse the backend
        VPIBackend backend;
  
        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backend = VPI_BACKEND_PVA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                      "' not recognized, it must be either cpu, cuda or pva.");
        }

        cvImage = cv::imread(strInputFileName);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        // =================================
         // Allocate all VPI resources needed
  
         // Create the stream where processing will happen
         CHECK_STATUS(vpiStreamCreate(0, &stream));
  
        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImage, 0, &imgInput));
  
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_S16, 0, &imgGrayscale));
  
        // Create the output keypoint array. Currently for PVA backend it must have 8192 elements.
        CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_KEYPOINT, 0, &keypoints));
  
        // Create the output scores array. It also must have 8192 elements and elements must be uint32_t.
        CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_U32, 0, &scores));
  
        // Create the payload for Harris Corners Detector algorithm
        CHECK_STATUS(vpiCreateHarrisCornerDetector(backend, cvImage.cols, cvImage.rows, &harris));
  
        // Define the algorithm parameters. We'll use defaults, expect for sensitivity.
        VPIHarrisCornerDetectorParams harrisParams;
        CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams));
        harrisParams.sensitivity = 0.01;

        // ================
        // Processing stage

        clock_t t;
        t = clock();
  
        // First convert input to grayscale
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgInput, imgGrayscale, NULL));
  
        // Then get Harris corners
        CHECK_STATUS(
            vpiSubmitHarrisCornerDetector(stream, backend, harris, imgGrayscale, keypoints, scores, &harrisParams));
  
        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        t = clock() - t;
        std::cout << "Time: " << ((float)t/(CLOCKS_PER_SEC)) << std::endl;
  
        // =======================================
        // Output processing and saving it to disk
  
        // Lock output keypoints and scores to retrieve its data on cpu memory
        VPIArrayData outKeypointsData;
        VPIArrayData outScoresData;
        VPIImageData imgData;
        CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ, &outKeypointsData));
        CHECK_STATUS(vpiArrayLock(scores, VPI_LOCK_READ, &outScoresData));
        CHECK_STATUS(vpiImageLock(imgGrayscale, VPI_LOCK_READ, &imgData));
  
        VPIKeypoint *outKeypoints = (VPIKeypoint *)outKeypointsData.data;
        uint32_t *outScores       = (uint32_t *)outScoresData.data;
  
        printf("\n%d keypoints found\n", *outKeypointsData.sizePointer);
  
        cv::Mat img;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgData, &img));
  
        cv::Mat outImage = DrawKeypoints(img, outKeypoints, outScores, *outKeypointsData.sizePointer);
  
        imwrite("harris_corners_" + strBackend + ".png", outImage);
  
        // Done handling outputs, don't forget to unlock them.
        CHECK_STATUS(vpiImageUnlock(imgGrayscale));
        CHECK_STATUS(vpiArrayUnlock(scores));
        CHECK_STATUS(vpiArrayUnlock(keypoints));
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up
  
    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }
  
    vpiImageDestroy(imgInput);
    vpiImageDestroy(imgGrayscale);
    vpiArrayDestroy(keypoints);
    vpiArrayDestroy(scores);
    vpiPayloadDestroy(harris);
    vpiStreamDestroy(stream);
  
    return retval;
}