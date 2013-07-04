#include <opencv2/core/core.hpp>

// Default LSD parameters 
// SIGMA_SCALE 0.6    - Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
// QUANT       2.0    - Bound to the quantization error on the gradient norm. 
// ANG_TH      22.5   - Gradient angle tolerance in degrees.
// LOG_EPS     0.0    - Detection threshold: -log10(NFA) > log_eps
// DENSITY_TH  0.7    - Minimal density of region points in rectangle.
// N_BINS      1024   - Number of bins in pseudo-ordering of gradient modulus.

// PI 
#ifndef M_PI
#define M_PI        CV_PI           // 3.14159265358979323846 
#endif
#define M_3_2_PI    (3 * CV_PI) / 2   // 4.71238898038  // 3/2 pi 
#define M_2__PI     2 * CV_PI         // 6.28318530718  // 2 pi 

// Label for pixels with undefined gradient. 
#define NOTDEF      double(-1024.0)

#define NOTUSED     0   // Label for pixels not used in yet. 
#define USED        1   // Label for pixels already used in detection. 

#define RELATIVE_ERROR_FACTOR 100.0

const double DEG_TO_RADS = M_PI / 180;

//const double NFA_ORIENT_THR = 0.01;

#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

struct edge
{
    cv::Point p;
    bool taken;
};
