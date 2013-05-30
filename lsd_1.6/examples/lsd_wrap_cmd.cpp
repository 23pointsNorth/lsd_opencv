#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "lsd_wrap.hpp"

using namespace std;
using namespace cv;
using namespace lsdwrap;

int main(int argc, char** argv)
{
	if (argc != 3) 
	{
		std::cout << "lsd_opencv_cmd [in] [out]" << std::endl
			<< "\tin - input image" << std::endl
			<< "\tout - output containing a line segment at each line [x1, y1, x2, y2, width, p, -log10(NFA)]" << std::endl;
		return false;
	}
	
	std::string in = argv[1];
	std::string out = argv[2];

	Mat image = imread(in, CV_LOAD_IMAGE_GRAYSCALE);
	
	LsdWrap lsd;
	vector<seg> segments;
	auto start = std::chrono::high_resolution_clock::now();
	
	lsd.lsdw(image, segments);
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	std::cout << segments.size() <<" line segments found. For " << duration << " ms." << std::endl;
	
	lsd.imshow_segs(string("Image"), image, segments);
	
	//Save to file
	ofstream segfile;
  	segfile.open(out);
	vector<seg>::iterator it = segments.begin(), eit = segments.end();
	for (; it!=eit; it++)
	{
		segfile << it->x1 << ' '
				<< it->y1 << ' '
				<< it->x2 << ' '
				<< it->y2 << ' '
				<< it->width << ' '
				<< it->p << ' '
				<< it->NFA << std::endl;
	}
	segfile.close();
	waitKey(0);
	
	return 0;
}
