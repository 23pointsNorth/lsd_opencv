/*----------------------------------------------------------------------------*/
/** @file lsd_opencv.h
    LSD OpenCV module header
    @author dani
 */
/*----------------------------------------------------------------------------*/

#ifndef LSD_OPENCV_H_
#define LSD_OPENCV_H_

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
#define M_3_2_PI    (3*CV_PI) / 2   // 4.71238898038  // 3/2 pi 
#define M_2__PI     2*CV_PI         // 6.28318530718  // 2 pi 

// Label for pixels with undefined gradient. 
#define NOTDEF      double(-1024.0)

#define NOTUSED     0   // Label for pixels not used in yet. 
#define USED        1   // Label for pixels already used in detection. 

#define RELATIVE_ERROR_FACTOR 100.0

const double DEG_TO_RADS = M_PI / 180;

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
/**
 * Create an LSD object. Specifying scale, number of subdivisions for the image, should the lines be refined and other constants as follows:
 *
 * @param _scale        The scale of the image that will be used to find the lines. Range (0..1].
 * @param _subdivision  The factor by which each dimension of the image will be divided into. 2 -> generates 2x2 rois and finds lines in them.
 *                      Note: Using smalled images (higher subdivision factor) will find fines lines.
 * @param _refine       Should the lines found be refined? E.g. breaking arches into smaller line approximations. 
 *                      If disabled, execution is faster.
 * @param _sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
 * @param _quant        Bound to the quantization error on the gradient norm. 
 * @param _ang_th       Gradient angle tolerance in degrees.
 * @param _log_eps      Detection threshold: -log10(NFA) > _log_eps
 * @param _density_th   Minimal density of aligned region points in rectangle.
 * @param _n_bins       Number of bins in pseudo-ordering of gradient modulus.
 */
    LSD(double _scale = 0.8, int _subdivision = 1, bool _refine = true, 
        double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, 
        double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);
/**
 * Detect lines in the input image with the specified ROI.
 *
 * @param _image    A grayscale(CV_8UC1) input image. 
 * @param _lines    Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                          Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end. 
 *                          Returned lines are strictly oriented depending on the gradient.
 * @param _roi      Return: ROI of the image, where lines are to be found. If specified, the returning 
 *                          lines coordinates are image wise.
 * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param prec      Return: Vector of precisions with which the lines are found.
 * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%. 
 *                          The bigger the value, logarithmically better the detection.
 *                              * 1 corresponds to 10 mean false alarms
 *                              * 0 corresponds to 1 mean false alarm
 *                              * 1 corresponds to 0.1 mean false alarms
 */
    void detect(const cv::InputArray& _image, cv::OutputArray& _lines, cv::Rect _roi = cv::Rect(),
                cv::OutputArray& width = cv::noArray(), cv::OutputArray& prec = cv::noArray(),
                cv::OutputArray& nfa = cv::noArray());

private:
    cv::Mat image;
    cv::Mat_<double> scaled_image;
    double *scaled_image_data;
    cv::Mat_<double> angles;     // in radians 
    double *angles_data;
    cv::Mat_<double> modgrad;
    double *modgrad_data;
    cv::Mat_<uchar> used;

    int img_width;
    int img_height;

    cv::Rect roi;
    int roix, roiy;

    const double SCALE;
    const bool doRefine;
    const int SUBDIVISION;
    const double SIGMA_SCALE;
    const double QUANT;
    const double ANG_TH;
    const double LOG_EPS;
    const double DENSITY_TH;
    const int N_BINS;

    struct RegionPoint {
        int x;
        int y;
        uchar* used;
        double angle;
        double modgrad;
    };

/**
 * Detect lines in the whole input image.
 *
 * @param _image        A grayscale(CV_8UC1) input image. 
 * @param lines         Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                              Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end. 
 *                              Returned lines are strictly oriented depending on the gradient.
 * @param widths        Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param precisions    Return: Vector of precisions with which the lines are found.
 * @param nfas          Return: Vector containing number of false alarms in the line region, with precision of 10%. 
 *                              The bigger the value, logarithmically better the detection.
 *                                  * 1 corresponds to 10 mean false alarms
 *                                  * 0 corresponds to 1 mean false alarm
 *                                  * 1 corresponds to 0.1 mean false alarms
 */
    void flsd(const cv::Mat_<double>& _image,
              std::vector<cv::Vec4i>& lines, 
              std::vector<double>* widths, std::vector<double>* precisions, 
              std::vector<double>* nfas);

/**
 * Finds the angles and the gradients of the image. Generates a list of pseudo ordered points.
 *
 * @param threshold The minimum value of the angle that is considered defined, otherwise NOTDEF
 * @param n_bins    The number of bins with which gradients are ordered by, using bucket sort. 
 * @param list      Return: Vector of coordinate points that are pseudo ordered by magnitude. 
 *                  Pixels would be ordered by norm value, up to a precision given by max_grad/n_bins.
 */
    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list);

/**
 * Grow a region starting from point s with a defined precision, 
 * returning the containing points size and the angle of the gradients.
 *
 * @param s         Starting point for the region.
 * @param reg       Return: Vector of points, that are part of the region
 * @param reg_size  Return: The size of the region.
 * @param reg_angle Return: The mean angle of the region.
 * @param prec      The precision by which each region angle should be aligned to the mean.
 */
    void region_grow(const cv::Point2i& s, std::vector<RegionPoint>& reg,
                     int& reg_size, double& reg_angle, const double& prec);

/**
 * Finds the bounding rotated rectangle of a region.
 *
 * @param reg       The region of points, from which the rectangle to be constructed from.
 * @param reg_size  The number of points in the region.
 * @param reg_angle The mean angle of the region.
 * @param prec      The precision by which points were found.
 * @param p         Probability of a point with angle within 'prec'.
 * @param rec       Return: The generated rectangle.
 */
    void region2rect(const std::vector<RegionPoint>& reg, const int reg_size, const double reg_angle,
                    const double prec, const double p, rect& rec) const;

/**
 * Compute region's angle as the principal inertia axis of the region.
 * @return          Regions angle.
 */
    double get_theta(const std::vector<RegionPoint>& reg, const int& reg_size, const double& x,
                     const double& y, const double& reg_angle, const double& prec) const;

/**
 * An estimation of the angle tolerance is performed by the standard deviation of the angle at points 
 * near the region's starting point. Then, a new region is grown starting from the same point, but using the 
 * estimated angle tolerance. If this fails to produce a rectangle with the right density of region points, 
 * 'reduce_region_radius' is called to try to satisfy this condition.
 */ 
    bool refine(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, const double& density_th);

/**
 * Reduce the region size, by elimination the points far from the starting point, until that leads to 
 * rectangle with the right density of region points or to discard the region if too small.
 */
    bool reduce_region_radius(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th);

/** 
 * Try some rectangles variations to improve NFA value. Only if the rectangle is not meaningful (i.e., log_nfa <= log_eps).
 * @return      The new NFA value.
 */
    double rect_improve();

/** 
 * Is the point at place 'address' aligned to angle theta, up to precision 'prec'?
 * @return      Whether the point is aligned.
 */
    bool isAligned(const int& address, const double& theta, const double& prec) const;


};

#endif /* !LSD_OPENCV_H_ */
