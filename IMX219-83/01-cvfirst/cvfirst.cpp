#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main (int argc, char **argv)
{
    Mat image, image_gray;
    image = imread(argv[1], IMREAD_COLOR );
    if (argc != 2 || !image.data) {
        cout << "No image data\n";
        return -1;
    }
   
    cvtColor(image, image_gray, COLOR_RGB2GRAY);
    namedWindow("image", WINDOW_AUTOSIZE);
    namedWindow("image gray", WINDOW_AUTOSIZE);
   
    imshow("image", image);
    imshow("image gray", image_gray);
   
    waitKey(0);
    return 0;
}
