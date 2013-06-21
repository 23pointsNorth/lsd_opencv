#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "lsd_opencv.hpp"

using namespace std;
using namespace cv;

#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720

int main(void)
{
    Mat img1(Size(IMAGE_WIDTH/2, IMAGE_HEIGHT), CV_8UC1, Scalar(255));
    Mat img2(Size(IMAGE_WIDTH/2, IMAGE_HEIGHT), CV_8UC1, Scalar(0));

    Mat img3(img1.size().height, img1.size().width + img2.size().width, CV_8UC1);
    Mat left(img3, Rect(0, 0, img1.size().width, img1.size().height));
    img1.copyTo(left);
    Mat right(img3, Rect(img1.size().width, 0, img2.size().width, img2.size().height));
    img2.copyTo(right);
    imshow("Image", img3);

    // LSD call 
    std::vector<Vec4i> lines;
    std::vector<double> width, prec, nfa;
    LSD ls;
    auto start = std::chrono::high_resolution_clock::now();
    

    ls.detect(img3, lines, Rect(), width, prec, nfa); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
    
    std::cout << lines.size() <<" line segments found. For " << duration << " ms." << std::endl;
    for (unsigned int i = 0; i < lines.size(); ++i)
    {
        cout << '\t' << "B: " << lines[i][0] << " " << lines[i][1] 
             << " E: " << lines[i][2] << " " << lines[i][3]
             << " W: " << width[i] 
             << " P:" << prec[i] 
             << " NFA:" << nfa[i] << std::endl;
    }

    waitKey(0);
    return 0;
}
