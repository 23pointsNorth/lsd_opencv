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

const int REPEAT_CYCLE = 5;

int main(int argc, char** argv)
{
	if (argc != 2) 
	{
		std::cout << "perf_test [in]" << std::endl
			<< "\tin - input image" << std::endl;
		return false;
	}
	
	std::string in = argv[1];

	Mat image = imread(in, CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << "Input image size: " << image.size() << std::endl;

	//
	// LSD 1.6 test
	//
	LsdWrap lsd_old;
	auto start = std::chrono::high_resolution_clock::now();
	for(unsigned int i = 0; i < REPEAT_CYCLE; ++i)
	{
		vector<seg> seg_old;
		lsd_old.lsdw(image, seg_old);
	}
	//lsd_old.lsd_subdivided(image, seg_old, 3);
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "lsd 1.6 - 1 cycle for avg. " << double(duration)/REPEAT_CYCLE << " ms." << std::endl;

	//
	// OpenCV LSD full test
	//
	LSD lsd_cv; // Refine founded lines
	start = std::chrono::high_resolution_clock::now();
	for(unsigned int i = 0; i < REPEAT_CYCLE; ++i)
	{
		vector<Vec4i> lines;
		lsd_cv.detect(image, lines);
	}
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "OpenCV full - 1 cycle for avg. " << double(duration)/REPEAT_CYCLE << " ms." << std::endl;
	
	//
	// OpenCV LSD not refined test
	//
	LSD lsd_notref(false); // Do not refine lines
	start = std::chrono::high_resolution_clock::now();
	for(unsigned int i = 0; i < REPEAT_CYCLE; ++i)
	{
		vector<Vec4i> lines;
		lsd_notref.detect(image, lines);
	}
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << "OpenCV not refined - 1 cycle for avg. " << double(duration)/REPEAT_CYCLE << " ms." << std::endl;
	
	return 0;
}
