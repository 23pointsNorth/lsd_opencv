#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
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
		std::cout << "lsd_opencv_cmd [in]" << std::endl
			<< "\tin - input image" << std::endl;
		return false;
	}
	
	std::string in = argv[1];

	Mat image = imread(in, CV_LOAD_IMAGE_GRAYSCALE);

	//
	// LSD 1.6 test
	//
	LsdWrap lsd_old;
	vector<seg> seg_old;
	auto start = std::chrono::high_resolution_clock::now();
	
	lsd_old.lsdw(image, seg_old);
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "lsd 1.6 \n \t" << seg_old.size() <<" line segments found. For " << duration << " ms." << std::endl;
	
	//
	// OpenCV LSD
	//
	LSD lsd_cv;
	vector<lineSegment> seg_cv;
	
	start = std::chrono::high_resolution_clock::now();
	lsd_cv.flsd(image, 0.8f, seg_cv);
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "OpenCV lsd \n \t" << seg_cv.size() <<" line segments found. For " << duration << " ms." << std::endl;

	//Copy new structure to old
	vector<seg> seg_cvo(seg_cv.size());
	for(unsigned int i = 0; i < seg_cvo.size(); ++i)
	{
		seg_cvo[i].x1 = seg_cv[i].begin.x;
		seg_cvo[i].y1 = seg_cv[i].begin.y;
		seg_cvo[i].x2 = seg_cv[i].end.x;
		seg_cvo[i].y2 = seg_cv[i].end.y;
		seg_cvo[i].p = seg_cv[i].p;
		seg_cvo[i].width = seg_cv[i].width;
		seg_cvo[i].NFA = seg_cv[i].NFA;
	}

	//
	// Show difference
	//
	Mat diff(image.size(), CV_8UC1, cv::Scalar(0));
	int d = lsd_old.CompareSegs(seg_old, seg_cvo, image.size(), string("Segment diff"), &diff);
	std::cout << "There are " << d << " none overlapping pixels" << std::endl;
	waitKey(0); // wait for human action 
	
	return 0;
}
