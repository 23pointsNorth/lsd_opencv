#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include "lsd_opencv.hpp"

#include "opencv2/opencv.hpp"
//core, imgproc

// LSD parameters 
#define SIGMA_SCALE 0.6    // Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
#define QUANT       2.0    // Bound to the quantization error on the gradient norm. 
#define ANG_TH      22.5   // Gradient angle tolerance in degrees.
#define LOG_EPS     0.0    // Detection threshold: -log10(NFA) > log_eps
#define DENSITY_TH  0.7    // Minimal density of region points in rectangle.
#define N_BINS      1024   // Number of bins in pseudo-ordering of gradient modulus.

// Other constants
// ln(10) 
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif // !M_LN10

// PI 
#ifndef M_PI
#define M_PI        CV_PI           // 3.14159265358979323846 
#endif
#define M_3_2_PI    (3*CV_PI) / 2   // 4.71238898038  // 3/2 pi 
#define M_2__PI     2*CV_PI         // 6.28318530718  // 2 pi 

// Label for pixels with undefined gradient. 
#define NOTDEF -1024.0

#define NOTUSED 0   // Label for pixels not used in yet. 
#define USED    1   // Label for pixels already used in detection. 

using namespace cv;

void LSD::flsd(const Mat& image, const double& scale, std::vector<Point2f>& begin, std::vector<Point2f>& end, 
    std::vector<double>& width, std::vector<double>& prec, std::vector<double>& nfa, Rect roi)
{
    //call the other method,
}

void LSD::flsd(const Mat& image, const double& scale, std::vector<lineSegment>& lines, Rect roi)
{
    CV_Assert(image.data != NULL);
    CV_Assert(scale > 0);

    // Angle tolerance
    double prec = M_PI * ANG_TH / 180.0;
    double p = ANG_TH / 180.0;
    double rho = QUANT / sin(prec);    // gradient magnitude threshold

    Mat angles;
    if (scale != 1)
    {
        Mat scaled_img;
        double sigma = (scale < 1.0)?(SIGMA_SCALE / scale):(SIGMA_SCALE);
        double prec = 3.0;
        unsigned int h =  (unsigned int) ceil(sigma * sqrt(2.0 * prec * log(10.0)));
        int ksize = 1+2*h; // kernel size 
        // create a Gaussian kernel
        Mat kernel = getGaussianKernel(ksize, sigma, CV_64F);
        // apply to the image
        filter2D(image, scaled_img, image.depth(), kernel, Point(-1, -1));
        imshow("Gaussian image", scaled_img);
        ll_angle(scaled_img, rho, angles);
    }
    else
    {
        ll_angle(image, rho, angles);
    }

}

void LSD::ll_angle(const Mat& in, const double& threshold, Mat& angles)
{

}
