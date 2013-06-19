/*----------------------------------------------------------------------------*/
/** @file lsd_opencv.h
    LSD OpenCV module header
    @author dani
 */
/*----------------------------------------------------------------------------*/

#ifndef LSD_OPENCV_H_
#define LSD_OPENCV_H_

#include <opencv2/core/core.hpp>

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


typedef struct lineSegment_s
{
    cv::Point2f begin;
    cv::Point2f end;
    double width;
    double p;
    double NFA;
    lineSegment_s(cv::Point2f _b, cv::Point2f _e, double _w, double _p, double _NFA)
      : begin(_b), end(_e), width(_w), p(_p), NFA(_NFA) { }
  } lineSegment;


typedef struct coorlist_s
{
  cv::Point2i p;
  struct coorlist_s* next;
} coorlist;

typedef struct rect_s
{
  double x1, y1, x2, y2;    /* first and second point of the line segment */
  double width;             /* rectangle width */
  double x, y;            /* center of the rectangle */
  double theta;             /* angle */
  double dx,dy;             /* (dx,dy) is vector oriented as the line segment */
  double prec;              /* tolerance angle */
  double p;                 /* probability of a point with angle within 'prec' */
} rect;

class LSD
{
public:
    void flsd(const cv::Mat& _image, const double& scale, 
              std::vector<cv::Point2f>& begin, std::vector<cv::Point2f>& end, 
              std::vector<double>& width, std::vector<double>& prec, 
              std::vector<double>& nfa, cv::Rect roi = cv::Rect());
    void flsd(const cv::Mat& _image, const double& scale, std::vector<lineSegment>& lines, 
              cv::Rect roi = cv::Rect());

private:
    cv::Mat image;
    cv::Mat scaled_image;
    double *scaled_image_data;
    cv::Mat angles;
    double *angles_data;
    cv::Mat modgrad;
    double *modgrad_data;

    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list);
    void region_grow(const cv::Point2i& s, std::vector<cv::Point2i>& reg, 
                     int& reg_size, double& reg_angle, double& prec, cv::Mat& used);
    void region2rect(const std::vector<cv::Point2i>& reg, const int reg_size, const double reg_angle, 
                    const double prec, const double p, rect& rec) const;
    bool refine(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                double prec, double p, rect& rec, const double& density_th, cv::Mat& used);
    bool reduce_region_radius(std::vector<cv::Point2i>& reg, int& reg_size, double reg_angle, 
                double prec, double p, rect& rec, double density, const double& density_th, cv::Mat& used);
    double dist(double x1, double y1, double x2, double y2);
    double rect_improve();
    bool isAligned(const int& address, const double& theta, const double& prec);
    double angle_diff(const double& a, const double& b) const;
    double angle_diff_signed(const double& a, const double& b) const;
    bool double_equal(const double& a, const double& b) const;
    double get_theta(const std::vector<cv::Point2i>& reg, const int& reg_size, const double& x, 
                     const double& y, const double& reg_angle, const double& prec) const;
};

#endif /* !LSD_OPENCV_H_ */
/*----------------------------------------------------------------------------*/
