 #include <opencv2/core/version.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
 #if CV_MAJOR_VERSION >= 3
 #    include <opencv2/imgcodecs.hpp>
 #else
 #    include <opencv2/highgui/highgui.hpp>
 #endif
  
 #include <iostream>
  
 // All vpi headers are under directory vpi/
 #include <vpi/OpenCVInterop.hpp>
  
 #include <vpi/Image.h>
 #include <vpi/Stream.h>
  
 // Algorithms we'll need:
 // - Image format conversion
 // - Box filter for blurring
 #include <vpi/algo/BoxFilter.h>
 #include <vpi/algo/ConvertImageFormat.h>
  
 int main(int argc, char *argv[])
 {
     if (argc != 2)
     {
         std::cerr << "Must pass an input image to be blurred" << std::endl;
         return 1;
     }
  
     // Phase 1: Initialization ---------------------------------
  
     // First load the input image
     cv::Mat cvImage = cv::imread(argv[1]);
     if (cvImage.data == NULL)
     {
         std::cerr << "Can't open input image" << std::endl;
         return 2;
     }
  
     // Now create the stream. Passing 0 allows algorithms submitted to it to run
     // in any available backend.
     VPIStream stream;
     vpiStreamCreate(0, &stream);
  
     // Now wrap the loaded image into a VPIImage object to be used by VPI.
     // VPI will deduce the image type based on how many channels cvImage has.
     // In this case, for OpenCV, which has 3 channels, it'll deduce to BGR8.
     VPIImage image;
     vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image);
  
     // Since we want to work with a grayscale version of input image, let's create
     // an image that will store it.
     VPIImage imageGray;
     vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imageGray);
  
     // Now create the output images, single unsigned 8-bit channel. The image lifetime is
     // managed by VPI.
     VPIImage blurred;
     vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &blurred);
  
     // Phase 2: main processing --------------------------------------
  
     // Pre-process the input, converting it from BGR8 to Grayscale
     vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, image, imageGray, NULL);
  
     // Submit the algorithm task for processing along with inputs and outputs
     // Parameters are:
     // 1. the stream on which the algorithm will run
     // 2. Which hardware backend will execute it. Here CUDA is specified, but it could be CPU or
     //    PVA (on Jetson Xavier devices). No further changes to the program are needed to have it executed
     //    on a different hardware.
     // 3. input image to be blurred.
     // 4. output image with the result.
     // 5 and 6. box filter size, in this case, 5x5
     // 7. Border extension for when the algorithm tries to sample pixels outside the image border.
     //    VPI_BORDER_ZERO will consider all such pixels as being 0.
     vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, imageGray, blurred, 5, 5, VPI_BORDER_ZERO);
  
     // Block the current thread until until the stream finishes all tasks submitted to it up till now.
     vpiStreamSync(stream);
  
     // Finally retrieve the output image contents and output it to disk
  
     // Lock output image to retrieve its data from cpu memory
     VPIImageData outData;
     vpiImageLock(blurred, VPI_LOCK_READ, &outData);
  
     // Construct an cv::Mat out of the retrieved data.
     cv::Mat cvOut;
     vpiImageDataExportOpenCVMat(outData, &cvOut);
  
     // Now write it to disk
     imwrite("tutorial_blurred.png", cvOut);
  
     // Done handling output image, don't forget to unlock it.
     vpiImageUnlock(blurred);
  
     // Stage 3: clean up --------------------------------------
  
     // Destroy all created objects.
     vpiStreamDestroy(stream);
     vpiImageDestroy(image);
     vpiImageDestroy(imageGray);
     vpiImageDestroy(blurred);
  
     return 0;
 }
  
 // vim: ts=8:sw=4:sts=4:et:ai

 