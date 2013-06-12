#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "lsd_opencv.hpp"

using namespace std;
using namespace cv;

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
	
	// LSD call 
	std::vector<lineSegment> lines;  
	LSD lsd;
    auto start = std::chrono::high_resolution_clock::now();
    lsd.flsd(image, 0.8f, lines); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
    
    std::cout << lines.size() <<" line segments found. For " << duration << " ms." << std::endl;
	
	//Save to file
	ofstream segfile;
  	segfile.open(out);
    for (unsigned int i = 0; i < lines.size(); ++i)
    {
        cout << '\t' << "B: "<< lines[i].begin << " E: " << lines[i].end << " W: " << lines[i].width 
             << " P:" << lines[i].p << " NFA:" << lines[i].NFA << std::endl;
		segfile << '\t' << "B: "<< lines[i].begin << " E: " << lines[i].end << " W: " << lines[i].width 
             << " P:" << lines[i].p << " NFA:" << lines[i].NFA << std::endl;
    }
	segfile.close();
	
	return 0;
}
