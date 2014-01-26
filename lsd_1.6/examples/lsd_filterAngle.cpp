#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include "lsd_opencv.hpp"

using namespace std;
using namespace cv;

const float ANGLE = 15;
const float RANGE = 10;

const float MIN_LEN = 30;

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
    std::vector<Vec4i> lines, filtered_lines, retained_lines, long_lines;
    std::vector<double> width, prec, nfa;
    Ptr<LineSegmentDetector> ls = createLineSegmentDetectorPtr(LSD_REFINE_STD);

    double start = double(getTickCount());
    ls->detect(image, lines);
    ls->filterSize(lines, lines, MIN_LEN, LSD_NO_SIZE_LIMIT);    // Remove all lines smaller than MIN_LEN pixels
    ls->filterOutAngle(lines,filtered_lines, ANGLE, RANGE);     // remove all vertical lines
    ls->retainAngle(lines, retained_lines, ANGLE, RANGE);       // take all vertical lines
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();

    cout << "It took " << duration_ms << " ms." << endl;

    //
    // Show difference
    //
    Mat drawnLines(image);
    ls->drawSegments(drawnLines, lines);
    imshow("Drawing segments", drawnLines);

    Mat vertical(image);
    ls->drawSegments(vertical, retained_lines);
    imshow("Retained lines", vertical);

    Mat difference = Mat::zeros(image.size(), CV_8UC3);
    int d = ls->compareSegments(image.size(), lines, filtered_lines, difference);
    imshow("Segments difference", difference);

    waitKey();
    return 0;
}
