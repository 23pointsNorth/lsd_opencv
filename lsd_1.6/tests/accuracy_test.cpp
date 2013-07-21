#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <lsd_opencv.hpp>

using namespace std;
using namespace cv;

const Size sz(640, 480);

void checkConstantColor()
{
	RNG rng(getTickCount());
	Mat constColor(sz, CV_8UC1, Scalar::all(rng.uniform(0, 256)));

	vector<Vec4i> lines;
	Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr();
	ls->detect(constColor, lines);

    Mat drawnLines = Mat::zeros(constColor.size(), CV_8UC1);
    ls->drawSegments(drawnLines, lines);
    imshow("checkConstantColor", drawnLines);

	std::cout << "Constant Color - Number of lines: " << lines.size() << " - 0 Wanted." << std::endl;
}

void checkWhiteNoise()
{
	//Generate white noise image
	Mat white_noise(sz, CV_8UC1);
	RNG rng(getTickCount());
	rng.fill(white_noise, RNG::UNIFORM, 0, 256);

	vector<Vec4i> lines;
	Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr();
	ls->detect(white_noise, lines);

	Mat drawnLines = Mat::zeros(white_noise.size(), CV_8UC1);
    ls->drawSegments(drawnLines, lines);
    imshow("checkRotatedRectangle", drawnLines);

	std::cout << "White Noise    - Number of lines: " << lines.size() << " - 0 Wanted." << std::endl;
}

void checkRotatedRectangle()
{
	RNG rng(getTickCount());
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
		line(filledRect, vertices[i], vertices[(i + 1) % 4], Scalar(255), 3);
	}

	vector<Vec4i> lines;
	Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr(LSD_REFINE_ADV);
	ls->detect(filledRect, lines);

    Mat drawnLines = Mat::zeros(filledRect.size(), CV_8UC1);
    ls->drawSegments(drawnLines, lines);
    imshow("checkRotatedRectangle", drawnLines);

	std::cout << "Check Rectangle- Number of lines: " << lines.size() << " - >= 4 Wanted." << std::endl;
}

void checkLines()
{
	RNG rng(getTickCount());
	Mat horzLines(sz, CV_8UC1, Scalar::all(rng.uniform(0, 128)));

	const int numLines = 3;
	for(unsigned int i = 0; i < numLines; ++i)
	{
		int y = rng.uniform(10, sz.width - 10);
		Point p1(y, 10);
		Point p2(y, sz.height - 10);
		line(horzLines, p1, p2, Scalar(255), 3);
	}

	vector<Vec4i> lines;
	Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr(LSD_REFINE_NONE);
	ls->detect(horzLines, lines);

	Mat drawnLines = Mat::zeros(horzLines.size(), CV_8UC1);
    ls->drawSegments(drawnLines, lines);
    imshow("checkLines", drawnLines);

	std::cout << "Lines Check   - Number of lines: " << lines.size() << " - " << numLines * 2 << " Wanted." << std::endl;
}

int main()
{
	checkWhiteNoise();
	checkConstantColor();
	checkRotatedRectangle();
	for (int i = 0; i < 10; ++i)
	{
		checkLines();
	}
	checkLines();
	cv::waitKey();
	return 0;
}