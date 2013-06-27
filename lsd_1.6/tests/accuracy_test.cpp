
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <lsd_opencv.hpp>

using namespace std;
using namespace cv;

const Size sz(640, 480);
	
void checkConstantColor()
{
	RNG rng;
	Mat constColor(sz, CV_8UC1, Scalar::all(rng.uniform(0, 256)));
	
	vector<Vec4i> lines;
	LSD ls;
	ls.detect(constColor, lines);
	
	LSD::showSegments("checkConstantColor", constColor, lines);
	
	std::cout << "Constant Color - Number of lines: " << lines.size() << " - 0 Wanted." << std::endl;
}

void checkWhiteNoise()
{
	//Generate white noise image
	Mat white_noise(sz, CV_8UC1);
	RNG rng;
	rng.fill(white_noise, RNG::UNIFORM, 0, 256);

	vector<Vec4i> lines;
	LSD ls;
	ls.detect(white_noise, lines);
	
	LSD::showSegments("checkWhiteNoise", white_noise, lines);
	
	std::cout << "White Noise    - Number of lines: " << lines.size() << " - 0 Wanted." << std::endl;
}

void checkRectangle()
{
	RNG rng;
	Mat filledRect = Mat::zeros(sz, CV_8UC1);
	
	Point center(rng.uniform(sz.width/4, sz.width*3/4),
				 rng.uniform(sz.height/4, sz.height*3/4));
	Size rect_size(rng.uniform(sz.width/8, sz.width/6),
				   rng.uniform(sz.height/8, sz.height/6));
	float angle = rng.uniform(0, 360);
	
	Point2f vertices[4];
	
	RotatedRect rRect = RotatedRect(center, rect_size, angle);

	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
	{
		line(filledRect, vertices[i], vertices[(i + 1) % 4], Scalar(255));
	}

	Rect brect = rRect.boundingRect();
	rectangle(filledRect, brect, Scalar(255));


	vector<Vec4i> lines;
	LSD ls;
	ls.detect(filledRect, lines);
	
	LSD::showSegments("checkConstantColor", filledRect, lines);
	
	std::cout << "Check Rectangle - Number of lines: " << lines.size() << " - 20 Wanted." << std::endl;
}

int main()
{
	checkWhiteNoise();
	checkConstantColor();
	checkRectangle();
	cv::waitKey(0);
	return 0;
}