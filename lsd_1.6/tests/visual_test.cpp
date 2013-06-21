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
	imshow("Input image", image);
	
	//
	// LSD 1.6 test
	//
	LsdWrap lsd_old;
	vector<seg> seg_old;
	auto start = std::chrono::high_resolution_clock::now();
	lsd_old.lsdw(image, seg_old);
	//lsd_old.lsd_subdivided(image, seg_old, 3);
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "lsd 1.6 \n \t" << seg_old.size() <<" line segments found. For " << duration << " ms." << std::endl;

	//
	// OpenCV LSD
	//
	// LSD lsd_cv(0.8, 1, false); // Do not refine lines
	LSD lsd_cv; // Refine founded lines
	vector<Vec4i> lines;
	
    std::vector<double> width, prec, nfa;
	start = std::chrono::high_resolution_clock::now();
	lsd_cv.detect(image, lines);
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "OpenCV lsd \n \t" << lines.size() <<" line segments found. For " << duration << " ms." << std::endl;
		
	//Copy new structure to old
	vector<seg> seg_cvo(lines.size());
	for(unsigned int i = 0; i < seg_cvo.size(); ++i)
	{
		seg_cvo[i].x1 = lines[i][0];
		seg_cvo[i].y1 = lines[i][1];
		seg_cvo[i].x2 = lines[i][2];
		seg_cvo[i].y2 = lines[i][3];
	}
	
	//
	// Show difference
	//
	Mat diff(image.size(), CV_8UC1, cv::Scalar(0));
	int d = lsd_old.CompareSegs(seg_old, seg_cvo, image.size(), string("Segment diff"), &diff);
	std::cout << "There are " << d << " not overlapping pixels" << std::endl;
	waitKey(0); // wait for human action 
	
	return 0;
}
