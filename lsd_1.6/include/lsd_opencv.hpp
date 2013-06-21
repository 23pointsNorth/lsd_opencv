/*----------------------------------------------------------------------------*/
/** @file lsd_opencv.h
    LSD OpenCV module header
    @author dani
 */
/*----------------------------------------------------------------------------*/

#ifndef LSD_OPENCV_H_
#define LSD_OPENCV_H_

#include <opencv2/core/core.hpp>

// // LSD parameters 
// #define SIGMA_SCALE 0.6    // Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
// #define QUANT       2.0    // Bound to the quantization error on the gradient norm. 
// #define ANG_TH      22.5   // Gradient angle tolerance in degrees.
// #define LOG_EPS     0.0    // Detection threshold: -log10(NFA) > log_eps
// #define DENSITY_TH  0.7    // Minimal density of region points in rectangle.
// #define N_BINS      1024   // Number of bins in pseudo-ordering of gradient modulus.

// Other constants
// ln(10) 
#ifndef M_LN10
#define M_LN10      2.30258509299404568402
#endif // !M_LN10

// PI 
#ifndef M_PI
#define M_PI        CV_PI           // 3.14159265358979323846 
#endif
#define M_3_2_PI    (3*CV_PI) / 2   // 4.71238898038  // 3/2 pi 
#define M_2__PI     2*CV_PI         // 6.28318530718  // 2 pi 

// Label for pixels with undefined gradient. 
#define NOTDEF      (double)-1024.0

#define NOTUSED     0   // Label for pixels not used in yet. 
#define USED        1   // Label for pixels already used in detection. 

#define BIN_SIZE    1024

#define RELATIVE_ERROR_FACTOR 100.0

typedef struct coorlist_s
{
    cv::Point2i p;
    struct coorlist_s* next;
} coorlist;

typedef struct rect_s
{
    double x1, y1, x2, y2;    /* first and second point of the line segment */
    double width;             /* rectangle width */
    double x, y;              /* center of the rectangle */
    double theta;             /* angle */
    double dx,dy;             /* (dx,dy) is vector oriented as the line segment */
    double prec;              /* tolerance angle */
    double p;                 /* probability of a point with angle within 'prec' */
} rect;

class LSD
{
public:
    LSD(double _scale = 0.8, int _subdivision = 1, bool _refine = true, 
        double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, 
        double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

    void detect(const cv::InputArray& _image, cv::OutputArray& _lines, cv::Rect roi = cv::Rect(),
                cv::OutputArray& width = cv::noArray(), cv::OutputArray& prec = cv::noArray(),
                cv::OutputArray& nfa = cv::noArray());

private:
    cv::Mat image;
    cv::Mat scaled_image;
    double *scaled_image_data;
    cv::Mat angles;
    double *angles_data;
    cv::Mat modgrad;
    double *modgrad_data;
    cv::Mat used;

    int img_width;
    int img_height;

    const double SCALE;
    const bool doRefine;
    const int SUBDIVISION;
    const double SIGMA_SCALE;
    const double QUANT;
    const double ANG_TH;
    const double LOG_EPS;
    const double DENSITY_TH;
    const int N_BINS;

    void flsd(const cv::Mat& _image,
              std::vector<cv::Vec4i>& lines, 
              std::vector<double>* widths, std::vector<double>* precisions, 
              std::vector<double>* nfas);
    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list);
    void region_grow(const cv::Point2i& s, std::vector<cv::Point2i>& reg, 
                     int& reg_size, double& reg_angle, const double& prec);
    void region2rect(const std::vector<cv::Point2i>& reg, const int reg_size, const double reg_angle, 
                    const double prec, const double p, rect& rec) const;
    bool refine(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                const double prec, double p, rect& rec, const double& density_th);
    bool reduce_region_radius(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                const double prec, double p, rect& rec, double density, const double& density_th);
    double dist(const double x1, const double y1, const double x2, const double y2) const;
    double rect_improve();
    bool isAligned(const int& address, const double& theta, const double& prec) const;
    double angle_diff(const double& a, const double& b) const;
    double angle_diff_signed(const double& a, const double& b) const;
    bool double_equal(const double& a, const double& b) const;
    double get_theta(const std::vector<cv::Point2i>& reg, const int& reg_size, const double& x, 
                     const double& y, const double& reg_angle, const double& prec) const;
};

#endif /* !LSD_OPENCV_H_ */
/*----------------------------------------------------------------------------*/
