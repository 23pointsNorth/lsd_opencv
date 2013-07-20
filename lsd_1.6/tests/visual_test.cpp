#include <iostream>
#include <fstream>
#include <string>

#include "lsd_wrap.hpp"
#include "lsd_opencv.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;
using namespace lsdwrap;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "visual_test [in]" << std::endl
			<< "\tin - input image" << std::endl;
		return false;
	}

	std::string in = argv[1];

	Mat image = imread(in, CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("Input image", image);

	//
	// LSD 1.6 test
	//
	LsdWrap lsd_old;
	vector<seg> seg_old;
	double start = double(getTickCount());
	lsd_old.lsdw(image, seg_old);
	//lsd_old.lsd_subdivided(image, seg_old, 3);
	double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	std::cout << "lsd 1.6 - blue\n\t" << seg_old.size() <<" line segments found. For " << duration_ms << " ms." << std::endl;

	//
	// OpenCV LSD
	//
	// LineSegmentDetector lsd_cv(LSD_NO_REFINE); // Do not refine lines
	LineSegmentDetector* lsd_cv = createLineSegmentDetectorPtr();
	vector<Vec4i> lines;

    std::vector<double> width, prec, nfa;
	start = double(getTickCount());

	std::cout << "Before" << std::endl;
	lsd_cv->detect(image, lines);
	std::cout << "After" << std::endl;

	duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	std::cout << "OpenCV lsd - red\n\t" << lines.size() <<" line segments found. For " << duration_ms << " ms." << std::endl;

	//Copy the old structure to the new
	vector<Vec4i> seg_cvo(seg_old.size());
	for(unsigned int i = 0; i < seg_old.size(); ++i)
	{
		seg_cvo[i][0] = seg_old[i].x1;
		seg_cvo[i][1] = seg_old[i].y1;
		seg_cvo[i][2] = seg_old[i].x2;
		seg_cvo[i][3] = seg_old[i].y2;
	}

	//
	// Show difference
	//
	Mat drawnLines(image);
	lsd_cv->drawSegments(drawnLines, lines);
	imshow("Drawing segments", drawnLines);

	Mat difference = Mat::zeros(image.size(), CV_8UC1);
	int d = lsd_cv->compareSegments(image.size(), seg_cvo, lines, &difference);
	imshow("Segments difference", difference);
	std::cout << "There are " << d << " not overlapping pixels." << std::endl;
	waitKey(0); // wait for human action

	return 0;
}
