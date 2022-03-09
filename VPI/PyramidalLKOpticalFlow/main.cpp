#include <opencv2/core/version.hpp>
 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

  
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>
  
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
  
#include <algorithm>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>
  
// Max number of corners detected by harris corner algo
constexpr int MAX_HARRIS_CORNERS = 8192;
  
// Max number of keypoints to be tracked
constexpr int MAX_KEYPOINTS = 100;
  
#define CHECK_STATUS(STMT)                                      \
    do                                                          \
    {                                                           \
        VPIStatus status__ = (STMT);                            \
        if (status__ != VPI_SUCCESS)                            \
        {                                                       \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];         \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));    \
            std::ostringstream ss;                              \
            ss << vpiStatusGetName(status__) << ": " << buffer; \
            throw std::runtime_error(ss.str());                 \
        }                                                       \
    } while (0);
  
static void SaveFileToDisk(VPIImage img, cv::Mat cvMask, std::string baseFileName, int32_t frameCounter)
{
    VPIImageData imgData;
    CHECK_STATUS(vpiImageLock(img, VPI_LOCK_READ, &imgData));
  
    cv::Mat cvImage;
    try
    {
        cv::Mat tmp;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgData, &tmp));
        cvtColor(tmp, cvImage, cv::COLOR_GRAY2BGR);
  
        CHECK_STATUS(vpiImageUnlock(img));
    }
    catch (...)
    {
        CHECK_STATUS(vpiImageUnlock(img));
        throw;
    }
  
    add(cvImage, cvMask, cvImage);
  
    // Create the output file name
    std::string fname = baseFileName;
    int ext           = fname.rfind('.');
  
    char buffer[512] = {};
    snprintf(buffer, sizeof(buffer) - 1, "%s_%04d%s", fname.substr(0, ext).c_str(), frameCounter,
            fname.substr(ext).c_str());
  
    // Finally, write frame to disk
    if (!imwrite(buffer, cvImage, {cv::IMWRITE_JPEG_QUALITY, 70}))
    {
        throw std::runtime_error("Can't write to " + std::string(buffer));
    }
}
  
// Sort keypoints by decreasing score, and retain only the first 'max'
static void SortKeypoints(VPIArray keypoints, VPIArray scores, int max)
{
    VPIArrayData ptsData, scoresData;
    CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ_WRITE, &ptsData));
    CHECK_STATUS(vpiArrayLock(scores, VPI_LOCK_READ_WRITE, &scoresData));
  
    std::vector<int> indices(*ptsData.sizePointer);
    std::iota(indices.begin(), indices.end(), 0);
  
    stable_sort(indices.begin(), indices.end(), [&scoresData](int a, int b) {
        uint32_t *score = reinterpret_cast<uint32_t *>(scoresData.data);
        return score[a] >= score[b]; // decreasing score order
    });
  
    // keep the only 'max' indexes.
    indices.resize(std::min<size_t>(indices.size(), max));
  
    VPIKeypoint *kptData = reinterpret_cast<VPIKeypoint *>(ptsData.data);
  
    // reorder the keypoints to keep the first 'max' with highest scores.
    std::vector<VPIKeypoint> kpt;
    std::transform(indices.begin(), indices.end(), std::back_inserter(kpt),
                    [kptData](int idx) { return kptData[idx]; });
    std::copy(kpt.begin(), kpt.end(), kptData);
  
    // update keypoint array size.
    *ptsData.sizePointer = kpt.size();
  
    vpiArrayUnlock(scores);
    vpiArrayUnlock(keypoints);
}
  
static int UpdateMask(cv::Mat &cvMask, const std::vector<cv::Scalar> &trackColors, VPIArray prevFeatures,
                       VPIArray curFeatures, VPIArray status)
{
    // Now that optical flow is completed, there are usually two approaches to take:
    // 1. Add new feature points from current frame using a feature detector such as
    //    \ref algo_harris_corners "Harris Corner Detector"
    // 2. Keep using the points that are being tracked.
    //
    // The sample app uses the valid feature point and continue to do the tracking.
  
    // Lock the input and output arrays to draw the tracks to the output mask.
    VPIArrayData curFeaturesData, statusData;
    CHECK_STATUS(vpiArrayLock(curFeatures, VPI_LOCK_READ_WRITE, &curFeaturesData));
    CHECK_STATUS(vpiArrayLock(status, VPI_LOCK_READ, &statusData));
  
    const VPIKeypoint *pCurFeatures = (VPIKeypoint *)curFeaturesData.data;
    const uint8_t *pStatus          = (uint8_t *)statusData.data;
  
    const VPIKeypoint *pPrevFeatures;
    if (prevFeatures)
    {
        VPIArrayData prevFeaturesData;
        CHECK_STATUS(vpiArrayLock(prevFeatures, VPI_LOCK_READ, &prevFeaturesData));
        pPrevFeatures = (VPIKeypoint *)prevFeaturesData.data;
    }
    else
    {
        pPrevFeatures = NULL;
    }
  
    int numTrackedKeypoints = 0;
    int totKeypoints        = *curFeaturesData.sizePointer;
  
    for (int i = 0; i < totKeypoints; i++)
    {
        // keypoint is being tracked?
        if (pStatus[i] == 0)
        {
            // draw the tracks
            cv::Point curPoint{(int)round(pCurFeatures[i].x), (int)round(pCurFeatures[i].y)};
            if (pPrevFeatures != NULL)
            {
                cv::Point2f prevPoint{pPrevFeatures[i].x, pPrevFeatures[i].y};
                line(cvMask, prevPoint, curPoint, trackColors[i], 2);
            }
  
            circle(cvMask, curPoint, 5, trackColors[i], -1);
  
            numTrackedKeypoints++;
        }
    }
  
    // We're finished working with the arrays.
    if (prevFeatures)
    {
        CHECK_STATUS(vpiArrayUnlock(prevFeatures));
    }
    CHECK_STATUS(vpiArrayUnlock(curFeatures));
    CHECK_STATUS(vpiArrayUnlock(status));
  
    return numTrackedKeypoints;
}
  
int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvFrame;
  
    // VPI objects that will be used
    VPIStream stream        = NULL;
    VPIImage imgTempFrame   = NULL;
    VPIImage imgFrame       = NULL;
    VPIPyramid pyrPrevFrame = NULL, pyrCurFrame = NULL;
    VPIArray prevFeatures = NULL, curFeatures = NULL, status = NULL;
    VPIPayload optflow = NULL;
    VPIArray scores    = NULL;
    VPIPayload harris  = NULL;
  
    int retval = 0;
  
    try
    {
        // ============================
        // Parse command line arguments
  
        if (argc != 5)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                      " <cpu|cuda> <input_video> <pyramid_levels> <output>");
        }
  
        std::string strBackend     = argv[1];
        std::string strInputVideo  = argv[2];
        int32_t pyrLevel           = std::stoi(argv[3]);
        std::string strOutputFiles = argv[4];
  
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
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu or cuda.");
        }
  
        {
            int ext        = strOutputFiles.rfind('.');
            strOutputFiles = strOutputFiles.substr(0, ext) + "_" + strBackend + strOutputFiles.substr(ext);
        }
  
        // ====================
        // Load the input video
        cv::VideoCapture invid;
        if (!invid.open(strInputVideo))
        {
            throw std::runtime_error("Can't open '" + strInputVideo + "'");
        }
  
        // Fetch the first frame and wrap it into a VPIImage.
        // The points to be tracked will be gathered from this frame later on.
        if (!invid.read(cvFrame))
        {
            throw std::runtime_error("Can't retrieve first frame from '" + strInputVideo + "'");
        }
  
        // =================================================
        // Allocate VPI resources and do some pre-processing
  
        // Create the stream where processing will happen.
        CHECK_STATUS(vpiStreamCreate(0, &stream));
  
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvFrame, 0, &imgTempFrame));
  
        // Create grayscale image representation of input.
        CHECK_STATUS(vpiImageCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, 0, &imgFrame));
  
        // Create the image pyramids used by the algorithm
        CHECK_STATUS(
            vpiPyramidCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5, 0, &pyrPrevFrame));
        CHECK_STATUS(vpiPyramidCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5, 0, &pyrCurFrame));
  
        // Create input and output arrays
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT, 0, &prevFeatures));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT, 0, &curFeatures));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status));
  
        // Create Optical Flow payload
        CHECK_STATUS(vpiCreateOpticalFlowPyrLK(backend, cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5,
                                                &optflow));
  
        // Parameters we'll use. No need to change them on the fly, so just define them here.
        // We're using the default parameters.
        VPIOpticalFlowPyrLKParams lkParams;
        CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(&lkParams));
  
        // Create a mask image for drawing purposes
        cv::Mat cvMask = cv::Mat::zeros(cvFrame.size(), CV_8UC3);
  
        // Gather feature points from first frame using Harris Corners on CPU.
        {
            CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores));
  
            VPIHarrisCornerDetectorParams harrisParams;
            CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams));
            harrisParams.strengthThresh = 0;
            harrisParams.sensitivity    = 0.01;
  
            CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CPU, cvFrame.cols, cvFrame.rows, &harris));
  
            // Convert input to grayscale to conform with harris corner detector restrictions
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgTempFrame, imgFrame, NULL));
  
            CHECK_STATUS(vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CPU, harris, imgFrame, curFeatures, scores,
                                                        &harrisParams));
  
            CHECK_STATUS(vpiStreamSync(stream));
  
            SortKeypoints(curFeatures, scores, MAX_KEYPOINTS);
        }
  
        // Create some random colors
        std::vector<cv::Scalar> trackColors;
        {
            std::vector<cv::Vec3b> tmpTrackColors;
  
            VPIArrayData ptsData;
            CHECK_STATUS(vpiArrayLock(curFeatures, VPI_LOCK_READ, &ptsData));
  
            const VPIKeypoint *pts = (VPIKeypoint *)ptsData.data;
  
            for (int i = 0; i < *ptsData.sizePointer; i++)
            {
                // Track hue depends on its initial position
                int hue = ((int)pts[i].x ^ (int)pts[i].y) % 180;
  
                tmpTrackColors.push_back(cv::Vec3b(hue, 255, 255));
            }
            CHECK_STATUS(vpiArrayUnlock(curFeatures));
  
            cvtColor(tmpTrackColors, tmpTrackColors, cv::COLOR_HSV2BGR);
  
            for (size_t i = 0; i < tmpTrackColors.size(); i++)
            {
                trackColors.push_back(cv::Scalar(tmpTrackColors[i]));
            }
        }
  
        // Update the mask with info from first frame.
        int numTrackedKeypoints = UpdateMask(cvMask, trackColors, NULL, curFeatures, status);
  
        // =================================================
        // Main processing stage
  
        // Generate pyramid for first frame.
        CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgFrame, pyrCurFrame));
  
        // Counter for the frames
        int idxFrame = 0;
  
        while (true)
        {
            // Save frame to disk
            SaveFileToDisk(imgFrame, cvMask, strOutputFiles, idxFrame);
  
            printf("Frame id=%d: %d points tracked. \n", idxFrame, numTrackedKeypoints);
  
            // Last iteration's current frame/features become this iteration's prev frame/features.
            // The former will contain information gathered in this iteration.
            std::swap(prevFeatures, curFeatures);
            std::swap(pyrPrevFrame, pyrCurFrame);
  
            // Fetch a new frame
            if (!invid.read(cvFrame))
            {
                printf("Video ended.\n");
                break;
            }
  
            ++idxFrame;
  
            // Wrap frame into a VPIImage, reusing the existing imgFrame.
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgTempFrame, cvFrame));
  
            // Convert it to grayscale
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backend, imgTempFrame, imgFrame, NULL))
  
            // Generate a pyramid out of it
            CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgFrame, pyrCurFrame));
  
            // Estimate the features' position in current frame given their position in previous frame
            CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream, 0, optflow, pyrPrevFrame, pyrCurFrame, prevFeatures,
                                                    curFeatures, status, &lkParams));
  
            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));
  
            // Update the output mask
            numTrackedKeypoints = UpdateMask(cvMask, trackColors, prevFeatures, curFeatures, status);
  
            // No more keypoints being tracked?
            if (numTrackedKeypoints == 0)
            {
                printf("No keypoints to track.\n");
                break; // we can finish procesing.
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }
  
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(harris);
    vpiPayloadDestroy(optflow);
  
    vpiPyramidDestroy(pyrPrevFrame);
    vpiImageDestroy(imgTempFrame);
    vpiImageDestroy(imgFrame);
    vpiArrayDestroy(prevFeatures);
    vpiArrayDestroy(curFeatures);
    vpiArrayDestroy(status);
    vpiArrayDestroy(scores);
  
    return retval;
}
  