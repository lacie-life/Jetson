#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>

int main( int argc, char** argv )
{
	// a command line argument containing the image filename is expected,
	// so make sure we have at least 2 args (the first arg is the program)
	if( argc < 2 )
	{
		printf("my-recognition:  expected image filename as argument\n");
		printf("example usage:   ./my-recognition my_image.jpg\n");
		return 0;
	}

	// retrieve the image filename from the array of command line args
	const char* imgFilename = argv[1];

	// these variables will store the image data pointer and dimensions
	uchar3* imgPtr = NULL;   // shared CPU/GPU pointer to image
	int imgWidth   = 0;      // width of the image (in pixels)
	int imgHeight  = 0;      // height of the image (in pixels)
		
	// load the image from disk as uchar3 RGB (24 bits per pixel)
	if( !loadImage(imgFilename, &imgPtr, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	// load the GoogleNet image recognition network with TensorRT
	// you can use imageNet::RESNET_18 to load ResNet-18 model instead
	imageNet* net = imageNet::Create(imageNet::GOOGLENET);

	// check to make sure that the network model loaded properly
	if( !net )
	{
		printf("failed to load image recognition network\n");
		return 0;
	}

	// this variable will store the confidence of the classification (between 0 and 1)
	float confidence = 0.0;

	// classify the image, return the object class index (or -1 on error)
	const int classIndex = net->Classify(imgPtr, imgWidth, imgHeight, &confidence);

	// make sure a valid classification result was returned	
	if( classIndex >= 0 )
	{
		// retrieve the name/description of the object class index
		const char* classDescription = net->GetClassDesc(classIndex);

		// print out the classification results
		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n", 
			  classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		// if Classify() returned < 0, an error occurred
		printf("failed to classify image\n");
	}
	
	// free the network's resources before shutting down
	delete net;
	return 0;
}