#include <opencv2/opencv.hpp>
#include <iostream>

class imageClass{

public:
    imageClass();
	~imageClass();

	cv::Mat image;
	std::string windowsName;
	int xi, xf, yi, yf;
	float pixelDistance;
	int click;
	bool distanciaReal;


private:
};