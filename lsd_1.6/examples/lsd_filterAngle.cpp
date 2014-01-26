#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include "lsd_opencv.hpp"

using namespace std;
using namespace cv;

#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "lsd_filter [in_image]" << std::endl
            << "\tin - input image" << std::endl;
        return false;
    }

    std::string in = argv[1];

    Mat image = imread(in, CV_LOAD_IMAGE_GRAYSCALE);

    //
    // LSD call
    //
    std::vector<Vec4i> lines, filtered_lines;
    std::vector<double> width, prec, nfa;
    Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr(LSD_REFINE_ADV);

    double start = double(getTickCount());
    ls->detect(image, lines, width, prec, nfa);
    ls->filterOutAngle(lines,filtered_lines, 90, 1); // remove all vertical lines
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();



    //
    // Show difference
    //
    Mat drawnLines(image);
    ls->drawSegments(drawnLines, lines);
    imshow("Drawing segments", drawnLines);

    Mat difference = Mat::zeros(image.size(), CV_8UC3);
    int d = ls->compareSegments(image.size(), lines, filtered_lines, difference);
    imshow("Segments difference", difference);

    waitKey();
    return 0;
}
