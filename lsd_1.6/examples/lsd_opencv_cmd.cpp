#include <iostream>
#include <fstream>
#include <string>
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
	std::vector<Vec4i> lines;
    std::vector<double> width, prec;
	Ptr<LineSegmentDetector> lsd = createLineSegmentDetectorPtr();

    double start = double(getTickCount());
    lsd->detect(image, lines, width, prec);
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();

    std::cout << lines.size() <<" line segments found. For " << duration_ms << " ms." << std::endl;

	//Save to file
	ofstream segfile;
  	segfile.open(out.c_str());
    for (unsigned int i = 0; i < lines.size(); ++i)
    {
		cout << '\t' << "B: " << lines[i][0] << " " << lines[i][1]
		<< " E: " << lines[i][2] << " " << lines[i][3]
		<< " W: " << width[i]
		<< " P:" << prec[i] << endl;
		segfile << '\t' << "B: " << lines[i][0] << " " << lines[i][1]
		<< " E: " << lines[i][2] << " " << lines[i][3]
		<< " W: " << width[i]
		<< " P:" << prec[i] << endl;
    }
	segfile.close();

	return 0;
}
